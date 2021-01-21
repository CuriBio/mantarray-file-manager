# -*- coding: utf-8 -*-

import os
import tempfile

from mantarray_file_manager import BasicWellFile
from mantarray_file_manager import CURRENT_HDF5_FILE_FORMAT_VERSION
from mantarray_file_manager import MantarrayH5FileCreator


def test_MantarrayH5FileCreator__sets_file_name_and_userblock_size_and_file_version():
    with tempfile.TemporaryDirectory() as tmp_dir:
        expected_filename = os.path.join(tmp_dir, "myfile.h5")
        test_file = MantarrayH5FileCreator(expected_filename)
        test_file.close()
        wf = BasicWellFile(expected_filename)
        test_file = wf.get_h5_file()
        assert test_file.userblock_size == 512
        assert test_file.filename == expected_filename
        assert (
            test_file.attrs["File Format Version"] == CURRENT_HDF5_FILE_FORMAT_VERSION
        )

        wf.get_h5_file().close()  # cleanup when running CI on windows systems