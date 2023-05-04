import math

import numpy as np
import polars as pl
from loguru import logger


def quant(ms_data: pl.DataFrame, pep: pl.DataFrame()) -> None:
    """Main function.

    Parameters
    ----------
    ms_data : pl.DataFrame
        Mass spec data file.
    pep : pl.DataFrame
        Peptide data file.
    """
    # TODO: protein quant
    # TODO: MAJOR OPTIMIZATION - I spent 2 days on this so there are LOTS
    # of ways top optimize
    # TODO: tests

    # Match the peptide to spectra
    pep_spectrum_df = match_peptide(ms_data, pep)
    # Perform peptide quant
    return peptide_quant(ms_data, pep_spectrum_df)


def match_peptide(ms_data: pl.DataFrame, pep: pl.DataFrame) -> pl.DataFrame:
    """Helper function to get the intensity of a peak at a given retention time.

    Parameters
    ----------
    ms_data : pl.DataFrame
        Mass spec data file.
    pep : pl.DataFrame
        Peptide data file.

    Returns
    -------
    pl.DataFrame:
        DataFrame with the matched peptide and mass spec data.
    """
    # Initialize the dictionary that will be converted into the returned dataframe
    data = {"peptide": [], "rt": [], "intensity": [], "mz": [], "ims": []}

    # Iterate through each row of the mass spec data and find the corresponding peptides
    # in the peptide dataframe.
    for row in ms_data.rows(named=True):
        # Convert the mzs to a series so they are easier to use
        row["mzs"] = pl.Series(row["mzs"])
        # Get all peptides at the given retention time
        peptide = pep.filter(pl.col("RetentionTime") == row["rts"])
        # For each peptide at the retention time, find the most intense peak and sum
        # across ion mobility
        for p in peptide.rows(named=True):
            # For each mz in the peptide mz list, get the mass spec mzs that are within
            # .02 of that mz. This `grouped_mz` list contains lists of indicies of mzs
            # from the mass spec data file that are close to the peptide mz.
            grouped_mzs = []
            for mz in p["mzs"]:
                mzs = list(
                    ((row["mzs"] <= mz + 0.02) & (row["mzs"] >= mz - 0.02)).arg_true()
                )
                # If those mzs have already been selected, continue. Otherwise add the
                # values to a list.
                if mzs in grouped_mzs:
                    continue
                else:
                    found = False
                    for group in grouped_mzs:
                        if all(x in group for x in mzs):
                            found = True
                            break
                    if not found:
                        grouped_mzs.append(mzs)
            # Initialize data to hold the information for the most intense peak
            most_intense = 0
            most_intense_mz = 0
            most_intense_ims = 0
            # Look at the mzs that are close to each other.
            for group in grouped_mzs:
                sum_intensity = 0
                total_mz = 0
                total_ims = 0
                num = 0
                # Sum mzs that are close to each other and average the mz and ims
                for ind in group:
                    if math.isclose(row["ims"][ind], p["IonMobility"], abs_tol=0.03):
                        sum_intensity += row["intensities"][ind]
                        total_mz += row["mzs"][ind]
                        total_ims += row["ims"][ind]
                        num += 1
                    if sum_intensity > most_intense:
                        most_intense = sum_intensity
                        most_intense_ims = total_ims / num
                        most_intense_mz = total_mz / num
            # Add the most intense peaks to the data
            data["peptide"].append(p["peptide"])
            data["intensity"].append(most_intense)
            data["rt"].append(row["rts"])
            data["mz"].append(most_intense_mz)
            data["ims"].append(most_intense_ims)

    # Return the polars dataframe with the peptide-spectrum matched data
    return pl.DataFrame(
        data,
        schema=[
            ("peptide", pl.Utf8),
            ("rt", pl.Float64),
            ("intensity", pl.Float64),
            ("mz", pl.Float64),
            ("ims", pl.Float64),
        ],
    )


def get_peaks(
    ms_data: pl.DataFrame, row: dict, i: int, next_val: float, right: float = True
) -> list:
    """Helper function to get corresponding peaks of a peptide at different rts.

    Parameters
    ----------
    ms_data : pl.DataFrame
        Mass spec data file.
    row : dict
        Row of the peptide data that contains the peptide we are interested in.
    i : int
        Index of the peptide of interest in the mass spec data file.
    next_val : float
        Intensity of next peak either to the right or the left.
        This help us to determine what part of the curve we are on (if we should stop
        as soon as the intensities increase/level off, or if the values should increase
        before they decrease).
    right : float, default = True
        True if we are looking at intensities to the right of the first measurement.
        False if we are looking to the left.

    Returns
    -------
    list:
        List containing the intensities of the matching peaks at different retention
        times. Another list of the retention times those peaks were found at.
    """
    # Initialize the lists that will contain the intensities/rts that we will return
    intensities = []
    rts = []

    # Keep track of the previous intensity so you know if the intensities are
    # decreasing or increasing. This could be simplified to just look at the
    # last item in the intensities list, but I'll do that later.
    last_val = row["intensity"]
    # Keep track of the number of times the intensity isn't changing (i.e. peak is
    # leveling off).
    count = 0

    # Here we determine if the peak is increasing or decreasing at this moment in time.
    # If it is decreasing, we should stop once the intensities increase or level off.
    # If it is increasing, we should see the intensities increase and then decrease.
    break_if_up = True
    if row["intensity"] < next_val:
        break_if_up = False

    # This is the loop that is getting all the intensities/rts for a peptide.
    while True:
        # If we are looking to the right of the matched intensity, make sure we are
        # in the range of our data and go 1 rt to the right. Otherwise, go 1 rt to
        # the left.
        if right:
            if i + 1 >= len(ms_data):
                return intensities, rts
            i += 1
        else:
            if i - 1 < 0:
                return intensities, rts
            i -= 1

        # Get the intensity and rt of the peak
        sum_intensity, rt = get_intensity_val(ms_data, row, i)

        # Return the data if any of the following criteria is met:
        #   1. the intensity is 0 of the current peak
        #   2. the intensity is increasing when it should be decreasing
        #   3. the intensity is leveling off
        if (
            sum_intensity == 0
            or (break_if_up and sum_intensity > last_val + last_val / 1000)
            or count >= 3
        ):
            return intensities, rts

        # If the intensity is decreasing when it should be, append the information to
        # our lists. Check if the values are roughly the same, meaning it is leveling
        # off.
        elif break_if_up and sum_intensity < last_val:
            intensities.append(sum_intensity)
            rts.append(rt)
            last_val = sum_intensity
            if sum_intensity > last_val + last_val / 1000:
                count += 1
            else:
                count = 0

        # If the intensity is increasing when it should be, append the information
        # to our lists.
        elif not break_if_up and sum_intensity > last_val:
            intensities.append(sum_intensity)
            rts.append(rt)
            last_val = sum_intensity
            count = 0

        # If the intensity is decreasing when it should be increasing, this means
        # we have gone over the top of our peak. Append the information to our
        # lists and set `break_if_up` to True.
        elif not break_if_up and sum_intensity < last_val:
            break_if_up = True
            intensities.append(sum_intensity)
            rts.append(rt)
            last_val = sum_intensity
            count = 0
        elif last_val == sum_intensity:
            count += 1
            if count >= 3:
                return intensities, rts
            intensities.append(sum_intensity)
            rts.append(rt)
        else:
            logger.info("Error: something weird happened")
            return


def get_intensity_val(ms_data: pl.DataFrame, row: dict, i: int) -> int:
    """Helper function to get the intensity of a peak at a given retention time.

    Parameters
    ----------
    ms_data : pl.DataFrame
            Mass spec data file.
    row : dict
        Row of the peptide data that contains the peptide we are interested in.
    i : int
            Index of the peptide of interest in the mass spec data file.

    Returns
    -------
    int:
        Integer containing the intensity of the peak.
        Integer containing the rt of the peak.
    """
    # Extract the relevant row from our mass spec data.
    curr_row = ms_data[i]

    # Get all mz values that are close to the peptide mz value.
    s = list(
        (
            (curr_row["mzs"].item() <= row["mz"] + 0.02)
            & (curr_row["mzs"].item() >= row["mz"] - 0.02)
        ).arg_true()
    )

    # Sum the intensities to collapse the ims dimention.
    sum_intensity = 0
    total_mz = 0
    num = 0
    for ind in s:
        if math.isclose(row["ims"], curr_row["ims"].item()[ind], abs_tol=0.03):
            sum_intensity += curr_row["intensities"].item()[ind]
            total_mz += curr_row["mzs"].item()[ind]
            num += 1
    return sum_intensity, curr_row["rts"].item()


def peptide_quant(ms_data: pl.DataFrame, matched_df: pl.DataFrame) -> pl.DataFrame:
    """Quantify peptides across rt.

    Parameters
    ----------
    ms_data : pl.DataFrame
        Mass spec data file.
    matched_df : pl.DataFrame
        Dataframe with peptide-spectrum matched data.

    Returns
    -------
    pl.DataFrame:
        DataFrame with the quantified peptides.
    """
    # Initialize the dictionary that will be converted into the returned dataframe
    pep_quant_data = {"peptide": [], "intensity": [], "mz": [], "num_fragments": []}
    # Add row numbers to the mass spec dataframe so I can easily go to the right
    # and left rts (this can be changed later to something more elegant)
    num_ms_data = ms_data.with_row_count()
    matched_df.to_pandas()
    num_ms_data.to_pandas()
    # Iterate through each row in the peptide-spectrum matched dataframe and
    # get the intensities for each peptide
    for row in matched_df.rows(named=True):
        # Get the row number in the mass spec dataframe for the current peptide
        i = num_ms_data.filter(pl.col("rts") == row["rt"])["row_nr"].item()
        # Get the intensities to the rt to the right and left of the current
        # retention time. This can probably be moved to a different function
        # or optimized later
        sum_intensity_left, _ = get_intensity_val(ms_data, row, i - 1)
        sum_intensity_right, _ = get_intensity_val(ms_data, row, i + 1)

        # Get the intensities/rts that match the peptide in rts to the left of
        # the current rt.
        i_left, rt_left = get_peaks(num_ms_data, row, i, sum_intensity_left, False)
        # These lists must be reversed since they will be added by going backward
        # in rt.
        if len(i_left) > 1:
            i_left, rt_left = i_left[::-1], rt_left[::-1]
        # Get the intensities/rts that match the peptide in rts to the right of
        # the current rt.
        i_right, rt_right = get_peaks(num_ms_data, row, i, sum_intensity_right, True)
        # Get a list with all intensities/rts for the peptide
        all_intensities = i_left + [row["intensity"]] + i_right
        rt_left + [row["rt"]] + rt_right
        # If there is more than one peak, do a trapezoidal integration
        if len(all_intensities) > 1:
            summed = np.trapz(
                all_intensities
            )  # tried to do `x=all_rts` but it gave me a weird answer
        # If there are no peaks, something went wrong.
        elif len(all_intensities) == 0:
            logger.info("Error: no peaks were found")
        # If there is only one peak, the intensity is the peak intensity
        else:
            summed = all_intensities[0]

        # Store the data
        pep_quant_data["peptide"].append(row["peptide"])
        pep_quant_data["intensity"].append(summed)
        pep_quant_data["mz"].append(row["mz"])
        pep_quant_data["num_fragments"].append(len(all_intensities))

    # Return the peptide quant dataframe
    return pl.DataFrame(
        pep_quant_data,
        schema=[
            ("peptide", pl.Utf8),
            ("intensity", pl.Float64),
            ("mz", pl.Float64),
            ("num_fragments", pl.Float64),
        ],
    )
