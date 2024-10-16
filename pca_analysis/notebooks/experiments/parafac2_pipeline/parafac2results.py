import duckdb as db
import polars as pl
import numpy as np
from tensorly.parafac2_tensor import apply_parafac2_projections, Parafac2Tensor
import plotly.graph_objects as go
import plotly.express as px
from pca_analysis.notebooks.experiments.parafac2_pipeline.utility import plot_imgs
from typing import Self
import logging
from pca_analysis.notebooks.experiments.parafac2_pipeline.data import XX
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def apply_projections(
    parafac2tensor: Parafac2Tensor,
) -> tuple[NDArray, list[NDArray], NDArray]:
    """
    apply the tensorly `apply_parafac2_projections` to scale the Bs

    :param parafac2tensor: result of PARAFAC2 decomposition
    :type parafac2tensor: Parafac2Tensor
    :return: A, B slices and C
    :rtype: tuple[NDArray, list[NDArray], NDArray]
    """
    weights, (A, B, C), projections = parafac2tensor
    _, (_, Bs, _) = apply_parafac2_projections((weights, (A, B, C), projections))

    return A, Bs, C


class Parafac2Results:
    def __init__(
        self, decomp: Parafac2Tensor, con: db.DuckDBPyConnection = db.connect()
    ) -> None:
        """
        generate a sql datamart for the decomposition results. After initialisation call
        `.ceate_datamart()` to establish the base state then use the various `viz` and
        `get` attributes to view and access the results.

        :param decomp: the decomposition result
        :type decomp: Parafac2Tensor
        :param con: conn to the db containing the results, defaults to db.connect()
        :type con: db.DuckDBPyConnection, optional
        """

        self._con = con
        self._decomp = decomp

        self._A, self._Bs, self._C = apply_projections(self._decomp)

        self.n_components = self._A.shape[1]

    def _create_component_table(self):
        """
        create a 'components' table containing the 'component' primary keys only.

        This is referenced by all other tables containing a 'component' column.
        """

        logger.debug("writing component table to db..")
        component_df = pl.DataFrame({"component": np.arange(0, self.n_components, 1)})
        component_df.shape

        self._con.execute(
            """--sql
            create table components (
                component integer primary key
            );
            insert into components
                select
                    component
                from
                    component_df
        """
        )

    def _create_table_A(self):
        """
        create a table for A - the weights of the components arranged with samples as
        rows and components as columns.
        """

        logger.debug("writing table A..")
        A_df = (
            pl.DataFrame(
                self._A, schema=[str(x) for x in np.arange(0, self.n_components)]
            )
            .with_row_index("sample")
            .unpivot(index=["sample"], variable_name="component", value_name="weight")
        )
        A_df.shape

        query = """--sql
        create table A (
        sample int references samples(sample),
        component int references components(component),
        weight float not null,
        primary key (sample, component)
        );
        insert into A
            select
                sample,
                component,
                weight
            from
                A_df
        """

        self._con.execute(query)

    def _create_sample_table(self, runids: list[str]):
        """write a table containing the unique sample ids
        :param runids: the labels of each sample in the dataset
        :type runids: list[str]
        """

        logger.debug("writing sample table..")
        sample_df = pl.DataFrame(
            {"sample": np.arange(0, len(runids), 1), "runid": runids}
        )
        sample_df.shape

        self._con.execute("""--sql
                          create table samples (
                          sample int,
                          runid varchar,
                          primary key (sample)
                          );
                          insert into samples
                            select
                                sample,
                                runid
                            from
                                sample_df
                          """)

    def _create_table_B(self):
        """
        Create a table containing the elution profile of each sample stacked samplewise
        """

        logger.debug("writing table B..")
        B_dfs = []

        for idx, b in enumerate(self._Bs):
            df = (
                pl.DataFrame(b)
                .with_columns(pl.lit(idx).alias("sample"))
                .with_row_index("elution_point")
                .unpivot(
                    index=["sample", "elution_point"],
                    variable_name="component",
                    value_name="value",
                )
                .with_columns(pl.col("component").str.replace("column_", "").cast(int))
            )
            B_dfs.append(df)
        B_df = pl.concat(B_dfs)

        try:
            query = """--sql
            create table B_pure (
            sample integer not null references samples(sample),
            component int not null references components(component),
            elution_point int not null,
            value float not null,
            primary key (sample, component, elution_point)
            );
            
            insert into B_pure
                select
                    sample,
                    component,
                    elution_point,
                    value
                from
                    B_df
            """
        except db.ConstraintException as e:
            description_of_index_cols = B_df.select(
                "sample", "elution_point", "component"
            ).describe()

            e.add_note(str(description_of_index_cols))
            raise e

        self._con.execute(query)

    def _create_table_C(self):
        """create the C table, the spectral profile of the components"""
        logger.debug("writing table C..")
        # create the df with an index, long
        C_df = (
            pl.DataFrame(self._C)
            .with_row_index("spectral_point")
            .unpivot(
                index="spectral_point", variable_name="component", value_name="value"
            )
            .with_columns(pl.col("component").str.replace("column_", "").cast(int))
        )
        C_df.shape
        # insert it into the db

        self._con.execute("""--sql
        create table C_pure (
                          component integer not null references components(component),
                          spectral_point int not null,
                          value float not null,
                          primary key (component, spectral_point)
        );
        insert into C_pure
            select
                component,
                spectral_point,
                value
            from
                C_df
        """)

    def _construct_component_tensors(
        self, A: NDArray, Bs: list[NDArray], C: NDArray
    ) -> list[NDArray]:
        """
        for the weights `A`, samplewise elution profiles `Bs` and spectral profile `C`
        create a three mode tensor of each component slice of each sample. Thus accessing
        the component mode will give you the image of that component for each sample.

        - A the weight of each component in each sample with shape: (samples, components)
        - Bs are a list of elution profiles of each component of each sample with shape:
        samples, (elution points, components)
        - C is the spectral profile of each component with shape (spectral points,
        components)

        :param A: The weights of the components per sample.
        :type A: NDArray
        :param Bs: The samplewise component elution profiles
        :type Bs: list[NDArray]
        :return: list of 3 mode tensors sample: (components, elution, spectral)
        :rtype: list[NDArray]
        """
        # construct each component of each sample as a np arr

        component_tensors = []
        for sample_idx, B in enumerate(Bs):
            tensor = np.einsum("ik,jk->kij", B * A[sample_idx], C)
            component_tensors.append(tensor)

        return [np.asarray(x) for x in component_tensors]

    def _create_component_tensor_df(
        self, slice: NDArray, sample_idx: int, component_idx: int
    ) -> pl.DataFrame:
        """
        create a component specific slice for a given sample_idx and component_idx

        :param slice: the component image
        :type slice: NDArray
        :param sample_idx: the sample index of the sample
        :type sample_idx: int
        :param component_idx: the component index of the component
        :type component_idx: int
        :raises ValueError: if duplicate labels exist in the normalised table
        :return: a normalised component dataframe
        :rtype: pl.DataFrame
        """
        df = (
            pl.DataFrame(slice)
            .with_columns(
                pl.lit(sample_idx).alias("sample"),
                pl.lit(component_idx).alias("component"),
            )
            .with_row_index("elution_point")
            .unpivot(
                index=["sample", "component", "elution_point"],
                variable_name="wavelength_point",
                value_name="abs",
            )
            .with_columns(
                pl.col("wavelength_point").str.replace("column_", "").cast(int)
            )
            .sort(["sample", "component", "wavelength_point", "elution_point"])
            .select(
                [
                    "sample",
                    "component",
                    "wavelength_point",
                    "elution_point",
                    "abs",
                ]
            )
        )

        dups = df.select(
            "sample", "component", "wavelength_point", "elution_point"
        ).is_duplicated()

        if dups.any():
            raise ValueError(
                f"duplicate primary key entry detected in sample: {sample_idx}, component: {component_idx}:{df.filter(dups)}"
            )
        return df

    def _create_component_slices_table(self):
        """create a sample x component slices table, where each row is an observation in
        time and spectral dimension of a component for a sample"""

        logger.debug("writing component slices..")
        component_tensors = self._construct_component_tensors(
            A=self._A, Bs=self._Bs, C=self._C
        )
        # flatten into a dataframe
        flat_dfs = []
        for sample_idx, tensor in enumerate(component_tensors):
            for component_idx, slice in enumerate(tensor):
                # check for duplicates in primary key

                flat_dfs.append(
                    self._create_component_tensor_df(slice, sample_idx, component_idx)
                )

        component_df = pl.concat(flat_dfs)
        len(component_df)  # quiet pylance

        # load into db
        self._con.sql("""--sql
        create table sample_components (
            sample int references samples(sample),
            component int references components(component),
            wavelength_point int not null,
            elution_point int not null,
            abs float not null,
            primary key (sample, component, wavelength_point, elution_point)
        );
        insert into sample_components
            select
                sample,
                component,
                wavelength_point,
                elution_point,
                abs
            from
                component_df
        """)

    def _create_recon_slices(self):
        """
        aggregate sum of sample component slices. This could be computed directly from
        the tensors however enforcing a linear dependency chain makes for simpler debugging
        """

        logger.debug("writing reconstructed slices..")

        # join all the tables together and sum. This can be done by pivoting on component

        self._con.sql(
            """--sql
        create or replace table sample_recons (
        sample int references samples(sample),
        wavelength_point int not null,
        elution_point int not null,
        abs float not null,
        primary key (sample, wavelength_point, elution_point)
        );
        with
            piv as (
            pivot
                sample_components
            on
                component
            using
                first(abs)
            order by
                sample,
                wavelength_point,
                elution_point
            )
            insert into sample_recons
                select
                    sample,
                    wavelength_point,
                    elution_point,
                    -- horizontal sum
                    list_sum(list_value(*columns(* exclude(sample, wavelength_point, elution_point)))) as abs 
                from
                    piv
        """
        )

    def create_datamart(self, input_imgs: XX) -> Self:
        """setup the base state of the data mart

        :param input_imgs: the input images, each sample as a slice.
        :type input_imgs: XX
        """

        logger.debug("creating datamart")

        self._create_component_table()
        self._create_sample_table(runids=input_imgs.runids)
        self._create_table_A()
        self._create_table_B()
        self._create_table_C()
        self._load_input_images(imgs=input_imgs)
        self._create_component_slices_table()
        self._create_recon_slices()

        return self

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
                    variable_name="wavelength_point",
                    value_name="abs",
                )
                .with_columns(
                    pl.lit(sample_idx).alias("sample"),
                    pl.col("wavelength_point").str.replace("column_", "").cast(int),
                )
            )

        input_img_df = pl.concat(dfs)
        len(input_img_df)
        # load into db
        self._con.execute(
            """--sql
        create table input_imgs (
            sample int references samples(sample),
            wavelength_point integer not null,
            elution_point integer not null,
            abs float not null,
            primary key (sample, wavelength_point, elution_point)
        );
        insert into input_imgs
            select
                sample,
                wavelength_point,
                elution_point,
                abs
            from
                input_img_df;
        """
        )

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

        df = self._con.execute(
            """--sql
            select
                sample,
                component,
                wavelength_point,
                elution_point,
                abs
            from
                sample_components
            where
                sample = ?
            and
                wavelength_point = ?
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

        df = self._con.execute(
            """--sql
            select
                sample,
                wavelength_point,
                elution_point,
                abs
            from
                input_imgs
            where
                sample = ?
            and
                wavelength_point = ?
            order by
                sample,
                wavelength_point,
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

        signals = self._con.execute(
            """--sql
            with
                joined as (
                    select
                        sample,
                        wavelength_point,
                        elution_point,
                        input.abs as input,
                        recon.abs as recon,
                    from
                        sample_recons as recon
                    join
                        input_imgs as input
                    using
                        (sample, wavelength_point, elution_point)
                    where
                    wavelength_point = ?
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
                sample, signal, wavelength_point, elution_point
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
        :facet_col: the column to facet on, either 'wavelength_point' or 'sample'.
        :return: faceted plot
        :rtype: go.Figure
        """

        if isinstance(samples, list):
            prepared_sample_cond = "sample in ?"
        elif isinstance(samples, int):
            prepared_sample_cond = "sample = ?"
        elif isinstance(samples, str) and (samples == "all"):
            prepared_sample_cond = ""

        if isinstance(wavelengths, list):
            prepared_wavelength_cond = "wavelengt_point in ?"
        elif isinstance(wavelengths, int):
            prepared_wavelength_cond = "wavelength_point = ?"
        elif isinstance(wavelengths, str) and (wavelengths == "all"):
            prepared_wavelength_cond = ""

        prefix = """--sql
        with
            joined as (
                select
                    sample,
                    wavelength_point,
                    elution_point,
                    input.abs as input,
                    recon.abs as recon
                from
                    sample_recons as recon
                join
                    input_imgs as input
                using (sample, wavelength_point, elution_point)
        """

        parameters = []

        if samples and wavelengths:
            parameters = [samples, wavelengths]
            conditions = " and ".join([prepared_sample_cond, prepared_wavelength_cond])
            conditions = " ".join(["where ", conditions])
        elif samples:
            parameters = [samples]
            conditions = " ".join(["where ", prepared_sample_cond])
        elif wavelengths:
            parameters = [wavelengths]
            conditions = " ".join(["where ", prepared_wavelength_cond])
        else:
            conditions = ""

        suffix = ") unpivot joined on input, recon into name signal value abs order by sample, signal, wavelength_point, elution_point;"

        query = "".join([prefix, conditions, suffix])

        df = self._con.execute(query, parameters=parameters).pl()

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
        return self._con.execute("select * from sample_recons").pl()

    def viz_recon_3d(self) -> go.Figure:
        """produce a 3d line plot of the full dataset, samples overlaid.

        :return: 3d line plot of the recon of a sample
        :rtype: go.Figure
        """

        df = self.get_normalised_recon()

        return df.pipe(
            plot_imgs,
            nm_col="wavelength_point",
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
            self._con.execute("select * from sample_recons")
            .pl()
            .pivot(
                index=["sample", "elution_point"], on="wavelength_point", values="abs"
            )
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

        return self._con.execute("show").pl()

    def _describe_table(self, table: str) -> pl.DataFrame:
        """display the description of a given `table`

        :param table: the table name
        :type table: str
        :return: tabulation of the description of the table
        :rtype: pl.DataFrame
        """
        return self._con.execute(f"describe {table}").pl()


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
