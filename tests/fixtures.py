# -*- coding: utf-8 -*-
import json
import os
from random import choice
from random import randint
import tempfile

from mantarray_file_manager import BaseWellFile
from mantarray_file_manager import Beta1WellFile
from mantarray_file_manager import H5Wrapper
from mantarray_file_manager import MAGNETOMETER_CONFIGURATION_UUID
from mantarray_file_manager import MantarrayH5FileCreator
from mantarray_file_manager import migrate_to_latest_version
from mantarray_file_manager import TIME_INDICES
from mantarray_file_manager import TIME_OFFSETS
from mantarray_file_manager import TISSUE_SAMPLING_PERIOD_UUID
from mantarray_file_manager import TISSUE_SENSOR_READINGS
from mantarray_file_manager import WellFile
from mantarray_file_manager.file_writer import h5_file_trimmer
import numpy as np
import pytest
from stdlib_utils import get_current_file_abs_directory

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()

PATH_TO_GENERIC_0_3_1_FILE = os.path.join(
    PATH_OF_CURRENT_FILE,
    "h5",
    "v0.3.1",
    "MA20123456__2020_08_17_145752__B3.h5",
)

PATH_TO_GENERIC_0_4_1_FILE = os.path.join(
    PATH_OF_CURRENT_FILE,
    "h5",
    "v0.4.1",
    "MA190190000__2021_01_19_011931__C3.h5",
)


@pytest.fixture(scope="module", name="current_beta1_version_file_path")
def fixture_current_beta1_version_file_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.4.2",
            "MA190190000__2021_01_19_011931__C3__v0.4.2.h5",
        )
        new_file_path = migrate_to_latest_version(file_path, tmp_dir)
        yield new_file_path


@pytest.fixture(scope="module", name="trimmed_file_path")
def fixture_trimmed_file_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.4.2",
            "MA190190000__2021_01_19_011931__C3__v0.4.2.h5",
        )
        new_file_path = migrate_to_latest_version(file_path, tmp_dir)
        trimmed_file_path = h5_file_trimmer(new_file_path, tmp_dir, 320, 320)
        yield trimmed_file_path


@pytest.fixture(scope="function", name="generic_beta_1_well_file")
def fixture_generic_beta_1_well_file():
    wf = Beta1WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "2020_08_04_build_775",
            "MA20001010__2020_08_04_220041__A3.h5",
        )
    )
    yield wf


@pytest.fixture(scope="function", name="generic_beta_1_well_file_0_3_1")
def fixture_generic_well_file_0_3_1():
    wf = Beta1WellFile(PATH_TO_GENERIC_0_3_1_FILE)
    yield wf


@pytest.fixture(scope="function", name="generic_h5_wrapper")
def fixture_generic_h5_wrapper():
    wrapper = H5Wrapper(PATH_TO_GENERIC_1_0_0_FILE)
    yield wrapper


@pytest.fixture(scope="function", name="generic_base_well_file")
def fixture_generic_base_well_file():
    bwf = BaseWellFile(PATH_TO_GENERIC_1_0_0_FILE)
    yield bwf


@pytest.fixture(scope="function", name="generic_beta_1_well_file_0_3_1__2")
def fixture_generic_well_file_0_3_1__2():
    wf = Beta1WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA20123456__2020_08_17_145752__A2.h5",
        )
    )
    yield wf


PATH_TO_GENERIC_1_0_0_FILE = os.path.join(
    PATH_OF_CURRENT_FILE, "beta_2_h5", "v1.0.0", "MA200440001__2021_05_24_212304__A1.h5"
)


@pytest.fixture(scope="module", name="current_beta2_version_file_path")
def fixture_current_beta2_version_file_path():
    yield PATH_TO_GENERIC_1_0_0_FILE


@pytest.fixture(scope="function", name="generic_beta_1_well_file_1_0_0")
def fixture_generic_well_file_1_0_0():
    wf = WellFile(PATH_TO_GENERIC_1_0_0_FILE)
    yield wf


@pytest.fixture(scope="function", name="well_file_1_0_0_with_random_config")
def fixture_well_file_1_0_0_with_random_config():
    # Tanner (6/9/21): Only metadata values needed for get_raw_channel_reading and get_tissue_sampling_period_microseconds are present in this file
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "random_file")
        h5_file = MantarrayH5FileCreator(file_path)

        # add random sampling period
        sampling_period_us = randint(1, 50) * 1000
        h5_file.attrs[str(TISSUE_SAMPLING_PERIOD_UUID)] = sampling_period_us
        # add magnetometer config and datasets
        magnetometer_config = dict()
        num_channels = 0
        for sensor in ("A", "B", "C"):
            axis_list = list()
            for axis in ("X", "Y", "Z"):
                # randomly add to config, but make sure at least one arbitrary channel is active
                if choice([False, True]) or (sensor == "C" and axis == "Z"):
                    axis_list.append(axis)
            # if no channels active for sensor, move on to the next one
            if not axis_list:
                continue
            magnetometer_config[sensor] = axis_list
            num_channels += len(axis_list)
        h5_file.attrs[str(MAGNETOMETER_CONFIGURATION_UUID)] = json.dumps(magnetometer_config)

        data_len = 100
        num_sensors = len(magnetometer_config.keys())
        tissue_data = np.arange(data_len * num_channels, dtype=np.int16)
        tissue_data.resize(num_channels, data_len)
        h5_file.create_dataset(
            TIME_INDICES,
            data=np.arange(1, data_len * sampling_period_us + 1, sampling_period_us, dtype=np.uint64),
        )
        h5_file.create_dataset(TIME_OFFSETS, data=np.ones((num_sensors, data_len), dtype=np.uint16))
        h5_file.create_dataset(TISSUE_SENSOR_READINGS, data=tissue_data)

        h5_file.close()

        wf = WellFile(file_path)
        yield wf
