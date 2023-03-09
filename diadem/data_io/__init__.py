from os import PathLike

from diadem.config import DiademConfig
from diadem.data_io.mzml import SpectrumStacker
from diadem.data_io.timstof import TimsSpectrumStacker


def read_raw_data(
    filepath: PathLike, config: DiademConfig
) -> TimsSpectrumStacker | SpectrumStacker:
    """Generic function to read data for DIA.

    It uses the file extension to know whether to dispatch the data to an
    mzML or timsTOF reader.
    """
    if str(filepath).endswith(".d") or str(filepath).endswith("hdf"):
        rf = TimsSpectrumStacker(filepath=filepath, config=config)
    elif str(filepath).lower().endswith(".mzml"):
        rf = SpectrumStacker(filepath, config=config)
    else:
        raise NotImplementedError
    return rf
