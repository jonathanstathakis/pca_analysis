from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

DbaseETLBase = declarative_base()


class Images(DbaseETLBase):
    __tablename__ = "images"
    runid: Mapped[str] = mapped_column(
        ForeignKey("chm.runid"),
        primary_key=True,
    )
    wavelength: Mapped[int] = mapped_column(primary_key=True)
    mins: Mapped[float] = mapped_column(primary_key=True)
    abs: Mapped[float] = mapped_column()


class Chm(DbaseETLBase):
    __tablename__ = "chm"
    runid: Mapped[str] = mapped_column(
        primary_key=True,
    )
    samplecode: Mapped[str]
    acq_date: Mapped[str]
    acq_method: Mapped[str]
    inj_vol: Mapped[float]
    seq_name: Mapped[str]
    vialnum: Mapped[str]
    originalfilepath: Mapped[str]
    id: Mapped[str]
    description: Mapped[str]
    path: Mapped[str]
