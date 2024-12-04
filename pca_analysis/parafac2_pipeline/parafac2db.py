import logging
from enum import StrEnum
from typing import Self

import duckdb as db
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from numpy.typing import NDArray
from sqlalchemy import Engine, ForeignKey, Sequence, text
from sqlalchemy.orm import Mapped, Session, mapped_column
from tensorly.parafac2_tensor import Parafac2Tensor

from .core_tables import ResultNames
from .data import XX
from .orm import ParafacResultsBase
from .utility import plot_imgs

pl.Config.set_tbl_width_chars(999)
logger = logging.getLogger(__name__)


class Components(ParafacResultsBase):
    __tablename__ = "components"
    component: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    exec_id: Mapped[str] = mapped_column(ForeignKey("exec_id.exec_id"))
    result_id: Mapped[str] = mapped_column(ForeignKey("result_id.result_id"))


class A(ParafacResultsBase):
    __tablename__ = "A"
    result_id: Mapped[str] = mapped_column(ForeignKey("result_id.result_id"))

    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"), primary_key=True
    )
    runid: Mapped[int] = mapped_column(
        ForeignKey("runids.runid"),
        primary_key=True,
    )
    component: Mapped[int] = mapped_column(
        ForeignKey("components.component"), primary_key=True
    )
    weight: Mapped[float] = mapped_column(nullable=False)


class Bs(ParafacResultsBase):
    __tablename__ = "b_pure"
    runid: Mapped[str] = mapped_column(ForeignKey("runids.runid"), primary_key=True)

    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"),
        primary_key=True,
    )
    result_id: Mapped[str] = mapped_column(
        ForeignKey("result_id.result_id"),
        primary_key=True,
    )
    component: Mapped[int] = mapped_column(
        ForeignKey("components.component"), primary_key=True
    )
    elution_point: Mapped[int] = mapped_column(primary_key=True)

    value: Mapped[float] = mapped_column(nullable=False)


class C(ParafacResultsBase):
    __tablename__ = "C_pure"
    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"), primary_key=True
    )
    result_id: Mapped[str] = mapped_column(
        ForeignKey("result_id.result_id"), primary_key=True
    )
    component: Mapped[int] = mapped_column(
        ForeignKey("components.component"), primary_key=True
    )
    wavelength: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[float] = mapped_column()


component_slices_row_idx = Sequence("component_slices_row_idx")


class ComponentSlices(ParafacResultsBase):
    __tablename__ = "component_slices"
    row_idx: Mapped[int] = mapped_column(
        component_slices_row_idx,
    )
    runid: Mapped[int] = mapped_column(ForeignKey("runids.runid"), primary_key=True)
    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"), primary_key=True
    )
    result_id: Mapped[str] = mapped_column(
        ForeignKey("result_id.result_id"), primary_key=True
    )
    component: Mapped[int] = mapped_column(
        ForeignKey("components.component"), primary_key=True
    )
    wavelength: Mapped[int] = mapped_column(primary_key=True)
    elution_point: Mapped[int] = mapped_column(primary_key=True)
    abs: Mapped[float] = mapped_column(nullable=False)


class SampleRecons(ParafacResultsBase):
    __tablename__ = "sample_recons"
    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"), primary_key=True
    )
    result_id: Mapped[str] = mapped_column(
        ForeignKey("result_id.result_id"), primary_key=True
    )
    runid: Mapped[int] = mapped_column(ForeignKey("runids.runid"), primary_key=True)
    wavelength: Mapped[int] = mapped_column(primary_key=True)
    elution_point: Mapped[int] = mapped_column(primary_key=True)
    abs: Mapped[float] = mapped_column(nullable=False)


class Parafac2Tables(StrEnum):
    A = "A"
    B_PURE = "B_pure"
    C_PURE = "C_pure"
    COMPONENTS = "components"
    SAMPLE_COMPONENTS = "component_slics"
    SAMPLE_RECONS = "sample_recons"


class Pfac2Loader:
    def __init__(
        self,
        exec_id: str,
        decomp: Parafac2Tensor,
        engine: Engine,
        runids: pl.DataFrame,
        wavelength_labels: list[int],
        results_name: str = "parafac2",
    ):
        """
        generate a sql datamart for the decomposition results. After initialisation call
        `.ceate_datamart()` to establish the base state then use the various `viz` and
        `get` attributes to view and access the results.

        :param decomp: the decomposition result
        :type decomp: Parafac2Tensor
        :param con: conn to the db containing the results, defaults to db.connect()
        :type con: db.DuckDBPyConnection, optional
        """
        self._exec_id = exec_id
        self._engine = engine
        self._decomp = decomp
        self._result_id = results_name
        self._runids = runids
        self._wavelength_labels = wavelength_labels
        self._A, self._Bs, self._C = apply_projections(self._decomp)

        self.n_components = self._A.shape[1]
        self.n_samples = self._A.shape[0]
        self.n_time_points = self._Bs[0].shape[0]
        self.n_wavelengths = self._C.shape[0]

    def __repr__(self):
        repr_str = f"""
        -------
        Results
        -------

        Parafac2Tensor
        --------------

        {str(self._decomp)}


        Factor Shapes
        -------------

        A: {self._A.shape}
        Bs: {len(self._Bs)}, {self._Bs[0].shape}
        C: {self._C.shape}

        Summary
        -------

        num. components: {self.n_components}
        num. samples: {self.n_samples}
        num. time points: {self.n_time_points}
        num. spectral points: {self.n_wavelengths}

        """

        return repr_str

    def _insert_into_result_id_tbl(self):
        """create a table storing the result ids"""

        logger.debug(f"adding {self._result_id} to results_id..")
        with Session(self._engine) as session:
            result_name = ResultNames(result_id=self._result_id, exec_id=self._exec_id)
            session.merge(result_name)
            session.commit()
        logger.debug("wrote to result_id.")

    def create_datamart(
        self,
    ) -> Self:
        """setup the base state of the data mart"""

        logger.debug("loading parafac2 results..")

        self._insert_into_result_id_tbl()
        self._create_component_table()
        self._create_table_A()
        self._create_table_B()
        self._create_table_C()
        self._create_component_slices_table()
        self._create_recon_slices()

        logger.debug("parafac2 results loaded.")

        return self

    def _create_component_table(self):
        """
        create a 'components' table containing the 'component' primary keys only.

        This is referenced by all other tables containing a 'component' column.
        """

        logger.debug("writing component table to db..")

        ParafacResultsBase.metadata.create_all(
            self._engine, tables=[Components.__table__]
        )

        with Session(self._engine) as session:
            for component in np.arange(0, self.n_components, 1):
                component = Components(
                    component=int(component),
                    exec_id=self._exec_id,
                    result_id=self._result_id,
                )

                session.merge(component)

            session.commit()

        logger.debug("written to component table.")

    def _create_table_A(self):
        """
        create a table for A - the weights of the components arranged with samples as
        rows and components as columns.
        """

        logger.debug("writing table A..")

        ParafacResultsBase.metadata.create_all(self._engine, tables=[A.__table__])

        with Session(self._engine) as session:
            for ss, sample in enumerate(self._A):
                for component, weight in enumerate(sample):
                    session.merge(
                        A(
                            exec_id=self._exec_id,
                            result_id=self._result_id,
                            runid=self._runids[ss],
                            component=component,
                            weight=weight,
                        )
                    )

            session.commit()

        logger.debug("A written to db.")

    def _create_table_B(self):
        """
        Create a table containing the elution profile of each sample stacked samplewise
        """

        ParafacResultsBase.metadata.create_all(self._engine, tables=[Bs.__table__])

        logger.debug("writing table B_pure..")

        with Session(self._engine) as session:
            for ss, sample in enumerate(self._Bs):
                for tt, time_point in enumerate(sample):
                    for component, abs in enumerate(time_point):
                        session.merge(
                            Bs(
                                runid=self._runids[ss],
                                exec_id=self._exec_id,
                                result_id=self._result_id,
                                component=component,
                                elution_point=tt,
                                value=abs,
                            )
                        )

            session.commit()

        logger.debug("Bs written to db..")

    def _create_table_C(self):
        """create the C table, the spectral profile of the components"""
        logger.debug("writing table C..")

        ParafacResultsBase.metadata.create_all(self._engine, tables=[C.__table__])

        with Session(self._engine) as session:
            for pp, point in enumerate(self._C):
                for cc, component in enumerate(point):
                    wavelength = self._wavelength_labels[pp]
                    c = C(
                        exec_id=self._exec_id,
                        result_id=self._result_id,
                        component=cc,
                        wavelength=wavelength,
                        value=component,
                    )
                session.merge(c)
            session.commit()

        logger.debug("C written to db..")

    def _create_component_slices_table(self):
        """create a sample x component slices table, where each row is an observation in
        time and spectral dimension of a component for a sample"""

        logger.debug("writing component slices..")
        component_tensors = self._construct_component_tensors(
            A=self._A, Bs=self._Bs, C=self._C
        )

        ParafacResultsBase.metadata.create_all(
            self._engine, tables=[ComponentSlices.__table__]
        )

        with Session(self._engine) as session:
            for sample_idx, tensor in enumerate(component_tensors):
                for component_idx, slice in enumerate(tensor):
                    for tt, time_point in enumerate(slice):
                        for wavelength_idx, abs in enumerate(time_point):
                            slice_row = ComponentSlices(
                                exec_id=self._exec_id,
                                result_id=self._result_id,
                                runid=self._runids[sample_idx],
                                component=component_idx,
                                wavelength=self._wavelength_labels[wavelength_idx],
                                elution_point=tt,
                                abs=abs,
                            )

                            session.merge(slice_row)
            session.commit()

        logger.debug("component sliced added to db.")

    def _create_recon_slices(self):
        """
        aggregate sum of sample component slices. This could be computed directly from
        the tensors however enforcing a linear dependency chain makes for simpler debugging
        """

        logger.debug("writing reconstructed slices..")

        # join all the tables together and sum. This can be done by pivoting on component

        ParafacResultsBase.metadata.create_all(
            self._engine, tables=[SampleRecons.__table__]
        )

        create_piv_tbl = f"""--sql
        create temporary table piv as (
            pivot
                (select
                    exec_id,
                    result_id,
                    runid,
                    component,
                    wavelength,
                    elution_point,
                    abs
                    from
                        {ComponentSlices.__tablename__}
                    )
            on
                component
            using
                first(abs)
            order by
                exec_id,
                result_id,
                runid,
                wavelength,
                elution_point
            );"""

        create_recon_tbl = """--sql
        create temporary table recon as (
                select
                    exec_id as exec_id,
                    runid as runid,
                    result_id as result_id,
                    wavelength as wavelength,
                    wavelength as wavelength,
                    elution_point as elution_point,
                    -- horizontal sum
                    list_sum(
                        list_value(
                            *columns(
                                * exclude(
                                    exec_id,
                                    result_id,
                                    runid,
                                    wavelength,
                                    elution_point
                                            )
                                        )
                                    )
                                ) as abs
                from
                    piv
            );
            """

        insert_into_sample_recons = f"""--sql
        insert or replace into {SampleRecons.__tablename__}
            select
                exec_id,
                result_id,
                runid,
                wavelength,
                elution_point,
                abs
            from
                recon;
        """

        with self._engine.connect() as conn:
            try:
                conn.begin()

                conn.execute(text(create_piv_tbl))
                conn.execute(text(create_recon_tbl))
                conn.execute(text(insert_into_sample_recons))
            except:
                conn.rollback()
                conn.close()
                raise
            else:
                conn.commit()
                conn.close()

    def _load_input_images(self, imgs: XX):
        """
        load the input images into the db for comparion with the reconstructions

        :param imgs: the input images, each sample as a slice.
        :type imgs: XX
        """
        logger.debug("writing input images..")
        # create a long dataframe

        dfs = []

        for sample_idx, img in enumerate(imgs):
            dfs.append(
                pl.DataFrame(img)
                .with_row_index("elution_point")
                .unpivot(
                    index=["elution_point"],
                    variable_name="wavelength",
                    value_name="abs",
                )
                .with_columns(
                    pl.lit(sample_idx).alias("sample"),
                    pl.col("wavelength").str.replace("column_", "").cast(int),
                )
            )

        input_img_df = pl.concat(dfs)
        len(input_img_df)
        # load into db
        self._conn.execute(
            """--sql
        create table input_imgs (
            sample int references samples(sample),
            wavelength integer not null,
            elution_point integer not null,
            abs float not null,
            primary key (sample, wavelength, elution_point)
        );
        insert into input_imgs
            select
                sample,
                wavelength,
                elution_point,
                abs
            from
                input_img_df;
        """
        )


class Parafac2Results:
    def __init__(self, conn: db.DuckDBPyConnection) -> None:
        self._conn = conn

    def _viz_overlay_components_sample_wavelength(
        self, sample: int, wavelength: int
    ) -> go.Figure:
        """for a given `sample` and `wavelength` provide an overlay plot of the
        components

        :param sample: the sample index
        :type sample: int
        :param wavelength: the wavelength index
        :type wavelength: int
        :return: plot of the overlay for the given `sample` and `wavelength`
        :rtype: go.Figure
        """

        # get the subset

        df = self._conn.execute(
            """--sql
            select
                sample,
                component,
                wavelength,
                elution_point,
                abs
            from
                sample_components
            where
                sample = ?
            and
                wavelength = ?
            """,
            parameters=[sample, wavelength],
        ).pl()

        return px.line(df, x="elution_point", y="abs", color="component")

    def _viz_input_img_curve_sample_wavelength(
        self, sample: int, wavelength: int
    ) -> go.Figure:
        """simple plot of an sample image curve for a given `sample` and `wavelength`

        :param sample: the sample index
        :type sample: int
        :param wavelength: the wavelength index
        :type wavelength: int
        :return: input image plot for a given `sample` and `wavelength`
        :rtype: go.Figure
        """

        df = self._conn.execute(
            """--sql
            select
                sample,
                wavelength,
                elution_point,
                abs
            from
                input_imgs
            where
                sample = ?
            and
                wavelength = ?
            order by
                sample,
                wavelength,
                elution_point
            """,
            parameters=[sample, wavelength],
        ).pl()

        return px.line(
            df,
            x="elution_point",
            y="abs",
            line_dash="sample",
            line_dash_map={"sample": "dash"},
            color_discrete_map={"sample": "orange"},
        )

    def viz_overlay_curve_components(self, sample: int, wavelength: int) -> go.Figure:
        """for a given `sample` and `wavelength` overlay the components and the input
        image curve

        :param sample: the sample index
        :type sample: int
        :param wavelength: the wavelength index
        :type wavelength: int
        :return: plot of the overlay for the given `sample` and `wavelength`
        :rtype: go.Figure
        """

        components = self._viz_overlay_components_sample_wavelength(
            sample=sample, wavelength=wavelength
        )
        img_curve = self._viz_input_img_curve_sample_wavelength(
            sample=sample, wavelength=wavelength
        )

        return go.Figure(data=components.data + img_curve.data)

    # def viz_overlay_input_recon(self, sample, wavelength)

    def viz_recon_input_overlay(self, sample, wavelength):
        """plot the reconstruction against the input curve for a given sample and wavelength"""

        signals = self._conn.execute(
            """--sql
            with
                joined as (
                    select
                        sample,
                        wavelength,
                        elution_point,
                        input.abs as input,
                        recon.abs as recon,
                    from
                        sample_recons as recon
                    join
                        input_imgs as input
                    using
                        (sample, wavelength, elution_point)
                    where
                    wavelength = ?
                    and
                        sample = ?)
            unpivot
                joined
            on
                input, recon
            into
                name signal
                value abs
            order by
                sample, signal, wavelength, elution_point
            """,
            parameters=[wavelength, sample],
        ).pl()

        fig = signals.pipe(px.line, x="elution_point", y="abs", color="signal")

        return fig

    def viz_recon_input_overlay_facet(
        self, samples=None, wavelengths=None, facet_col=None
    ) -> go.Figure:
        """produce a facet plot over a given dimension

        :param samples: the samples included in the plot. If "all", includes all samples,
        if an integer, returns that sample, if list[int], returns all those samples.
        :param wavelength: the wavelength included in the plot. If "all", includes all
        samples, if an integer, returns that sample, if list[int], returns all those
        wavelengths.
        :facet_col: the column to facet on, either 'wavelength' or 'sample'.
        :return: faceted plot
        :rtype: go.Figure
        """

        # if samples is a list return 'in' elif its a scalar return '=' else nothing
        if isinstance(samples, list):
            prepared_sample_cond = "sample in ?"
        elif isinstance(samples, int):
            prepared_sample_cond = "sample = ?"
        elif isinstance(samples, str) and (samples == "all"):
            prepared_sample_cond = ""

        if isinstance(wavelengths, list):
            prepared_wavelength_cond = "wavelength in ?"
        elif isinstance(wavelengths, int):
            prepared_wavelength_cond = "wavelength = ?"
        elif isinstance(wavelengths, str) and (wavelengths == "all"):
            prepared_wavelength_cond = ""

        prefix = """--sql
        with
            joined as (
                select
                    sample,
                    wavelength,
                    elution_point,
                    input.abs as input,
                    recon.abs as recon
                from
                    sample_recons as recon
                join
                    input_imgs as input
                using (sample, wavelength, elution_point)
        """

        parameters = []

        if samples and wavelengths:
            if samples == "all":
                parameters = [wavelengths]
                conditions = [prepared_wavelength_cond]
            else:
                parameters = [samples, wavelengths]
                conditions = [prepared_sample_cond, prepared_wavelength_cond]
            conditions = " ".join(["where ", *conditions])
        elif samples:
            parameters = [samples]
            conditions = " ".join(["where ", prepared_sample_cond])
        elif wavelengths:
            parameters = [wavelengths]
            conditions = " ".join(["where ", prepared_wavelength_cond])
        else:
            conditions = ""

        suffix = ") unpivot joined on input, recon into name signal value abs order by sample, signal, wavelength, elution_point;"

        query = "".join([prefix, conditions, suffix])

        df = self._conn.execute(query, parameters=parameters).pl()

        if df.is_empty():
            raise ValueError("df is empty")

        # calculate a facet col wrap
        facet_col_wrap = df.select(facet_col).unique().count().item() // 4

        fig = df.pipe(
            px.line,
            x="elution_point",
            y="abs",
            color="signal",
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
        )

        return fig

    def get_normalised_recon(self) -> pl.DataFrame:
        """
        return the fully normalised reconstruction as a polars dataframe. One row per
        observation.

        :return: normalised reconstruction table unfolded along every mode.
        :rtype: pl.DataFrame
        """
        return self._conn.execute("select * from sample_recons").pl()

    def viz_recon_3d(self) -> go.Figure:
        """produce a 3d line plot of the full dataset, samples overlaid.

        :return: 3d line plot of the recon of a sample
        :rtype: go.Figure
        """

        df = self.get_normalised_recon()

        return df.pipe(
            plot_imgs,
            nm_col="wavelength",
            abs_col="abs",
            time_col="elution_point",
            runid_col="sample",
        )

    def get_recon_df(self) -> pl.DataFrame:
        """return the

        :return: reconstructed data as a samplewise unfolded dataframe.
        :rtype: pl.DataFrame
        """

        return (
            self._conn.execute("select * from sample_recons")
            .pl()
            .pivot(index=["sample", "elution_point"], on="wavelength", values="abs")
            .sort(["sample", "elution_point"])
        )

    def recon_as_np_slices(self) -> list[NDArray]:
        """
        :return: the recon data as a list of slices, mimicking the tl implementation
        :rtype: list[NDArray]
        """

        recon_slices = (
            self.get_recon_df()
            .select(pl.exclude("elution_point"))
            .partition_by("sample", include_key=False, maintain_order=True)
        )
        recon_slices = [x.to_numpy(writable=True) for x in recon_slices]

        return recon_slices

    def recon_as_np_tensor(self) -> NDArray:
        """output the reconstruction as a three way numpy array with the first axis as
        samples, second as elution points and third as wavelengths

        :return: three mode tensor (samples, elution, wavelength)
        :rtype: NDArray
        """

        return np.stack(self.recon_as_np_slices())

    def _check_computations_match_tly(self) -> bool:
        """
        wrapper for `_proof_that_my_computations_match_tly`, a validation check.

        :return: True if difference is insignificant, else False.
        :rtype: bool
        """

        comp_tensor_list = self._construct_component_tensors(
            A=self._A, Bs=self._Bs, C=self._C
        )
        summed_ct_list = [np.sum(tensor, axis=0) for tensor in comp_tensor_list]

        return _proof_that_my_computations_match_tly(
            my_summed_ct_list=summed_ct_list, decomp=self._decomp
        )

    def _show_tables(self) -> pl.DataFrame:
        """display all tables in the database

        :return: tabular summary of the db tables
        :rtype: pl.DataFrame
        """

        return self._conn.execute("show").pl()

    def _describe_table(self, table: str) -> pl.DataFrame:
        """display the description of a given `table`

        :param table: the table name
        :type table: str
        :return: tabulation of the description of the table
        :rtype: pl.DataFrame
        """
        return self._conn.execute(f"describe {table}").pl()

    def results_dashboard(self):
        """return a Dash dashboard"""
        import dash_bootstrap_components as dbc
        from dash import Dash, dcc

        app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

        recon_3d = self.viz_recon_3d()
        facet = self.viz_recon_input_overlay_facet(samples="all", wavelengths=10)
        app.layout = dbc.Container(
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="3d", figure=recon_3d)),
                    dbc.Col(dcc.Graph(id="facet", figure=facet)),
                ]
            )
        )

        return app


def _proof_that_my_computations_match_tly(
    my_summed_ct_list: list[NDArray], decomp: Parafac2Tensor
):
    """
    My implementation of the postprocessing pipeline results in a marginal total difference
    to the tensorly results doing the same. The hypothesis is that the Polars/duckdb
    interface causes slight variance to emerge. This function circumvents the polars/duckdb
    section of the pipeline and checks whether the computed individual component tensors
    are correctly calculated. Returns True if the variance is less than 1x10^-10, meaning
    that the variance is insignificant, else False if the variance is greater, indicating that
    it is significant.

    :param my_summed_ct_list: reconstructed sample slices
    :type my_summed_ct_list: list[NDArray]
    :decomp: tensorly decomposition result
    :type decomp: Parafac2Tensor
    :return: True if difference is insignificant, else False
    :rtype: bool
    """

    from tensorly.parafac2_tensor import parafac2_to_tensor

    tly = parafac2_to_tensor(decomp)
    diff = np.sum(np.stack(my_summed_ct_list) - tly)

    print(diff)

    if np.log10(diff) < -10:
        return True
    else:
        return False


class Pfac2Extractor:
    def __init__(self):
        """extract parafac2 results from db"""


class PARAFAC2DB:
    def __init__(self, engine: Engine):
        """handler of the database IO for the PARAFAC2 results"""

        self.loader: Pfac2Loader
        self.extractor: Pfac2Extractor
        self._engine = engine

    def get_loader(
        self,
        exec_id: str,
        decomp: Parafac2Tensor,
        runids: list[str],
        wavelength_labels: list[int],
        results_name: str = "parafac2",
    ) -> Pfac2Loader:
        return Pfac2Loader(
            exec_id=exec_id,
            decomp=decomp,
            engine=self._engine,
            results_name=results_name,
            runids=runids,
            wavelength_labels=wavelength_labels,
        )

    def get_extractor(
        self,
        exec_id: str,
        results_con: db.DuckDBPyConnection,
        results_name: str = "parafac2",
    ) -> Pfac2Extractor:
        return Pfac2Extractor()

    def clear_tables(self):
        """clear all parafac2 related tables"""

        for tbl in Parafac2Tables:
            self._conn.execute(f"truncate {tbl}")
