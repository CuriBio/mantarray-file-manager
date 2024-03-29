# -*- coding: utf-8 -*-
import uuid

from mantarray_file_manager import ADC_GAIN_SETTING_UUID
from mantarray_file_manager import ADC_REF_OFFSET_UUID
from mantarray_file_manager import ADC_TISSUE_OFFSET_UUID
from mantarray_file_manager import BACKEND_LOG_UUID
from mantarray_file_manager import BARCODE_IS_FROM_SCANNER_UUID
from mantarray_file_manager import BOOTUP_COUNTER_UUID
from mantarray_file_manager import CENTIMILLISECONDS_PER_SECOND
from mantarray_file_manager import COMPUTER_NAME_HASH_UUID
from mantarray_file_manager import CURI_BIO_ACCOUNT_UUID
from mantarray_file_manager import CURI_BIO_USER_ACCOUNT_ID
from mantarray_file_manager import CURRENT_BETA1_HDF5_FILE_FORMAT_VERSION
from mantarray_file_manager import CURRENT_BETA2_HDF5_FILE_FORMAT_VERSION
from mantarray_file_manager import CUSTOMER_ACCOUNT_ID_UUID
from mantarray_file_manager import DATETIME_STR_FORMAT
from mantarray_file_manager import FILE_FORMAT_VERSION_METADATA_KEY
from mantarray_file_manager import FILE_MIGRATION_PATHS
from mantarray_file_manager import FILE_VERSION_PRIOR_TO_MIGRATION_UUID
from mantarray_file_manager import HARDWARE_TEST_RECORDING_UUID
from mantarray_file_manager import IS_FILE_ORIGINAL_UNTRIMMED_UUID
from mantarray_file_manager import MAGNETOMETER_CONFIGURATION_UUID
from mantarray_file_manager import MAIN_FIRMWARE_VERSION_UUID
from mantarray_file_manager import MANTARRAY_NICKNAME_UUID
from mantarray_file_manager import MANTARRAY_SERIAL_NUMBER_UUID
from mantarray_file_manager import METADATA_UUID_DESCRIPTIONS
from mantarray_file_manager import MICROSECONDS_PER_CENTIMILLISECOND
from mantarray_file_manager import MIN_SUPPORTED_FILE_VERSION
from mantarray_file_manager import NOT_APPLICABLE_H5_METADATA
from mantarray_file_manager import ORIGINAL_FILE_VERSION_UUID
from mantarray_file_manager import PCB_SERIAL_NUMBER_UUID
from mantarray_file_manager import PLATE_BARCODE_UUID
from mantarray_file_manager import REF_SAMPLING_PERIOD_UUID
from mantarray_file_manager import REFERENCE_SENSOR_READINGS
from mantarray_file_manager import REFERENCE_VOLTAGE_UUID
from mantarray_file_manager import SLEEP_FIRMWARE_VERSION_UUID
from mantarray_file_manager import SOFTWARE_BUILD_NUMBER_UUID
from mantarray_file_manager import SOFTWARE_RELEASE_VERSION_UUID
from mantarray_file_manager import START_RECORDING_TIME_INDEX_UUID
from mantarray_file_manager import STIMULATION_PROTOCOL_UUID
from mantarray_file_manager import STIMULATION_READINGS
from mantarray_file_manager import TAMPER_FLAG_UUID
from mantarray_file_manager import TIME_INDICES
from mantarray_file_manager import TIME_OFFSETS
from mantarray_file_manager import TISSUE_SAMPLING_PERIOD_UUID
from mantarray_file_manager import TISSUE_SENSOR_READINGS
from mantarray_file_manager import TOTAL_WELL_COUNT_UUID
from mantarray_file_manager import TOTAL_WORKING_HOURS_UUID
from mantarray_file_manager import TRIMMED_TIME_FROM_ORIGINAL_END_UUID
from mantarray_file_manager import TRIMMED_TIME_FROM_ORIGINAL_START_UUID
from mantarray_file_manager import USER_ACCOUNT_ID_UUID
from mantarray_file_manager import UTC_BEGINNING_DATA_ACQUISTION_UUID
from mantarray_file_manager import UTC_BEGINNING_RECORDING_UUID
from mantarray_file_manager import UTC_BEGINNING_STIMULATION_UUID
from mantarray_file_manager import UTC_FIRST_REF_DATA_POINT_UUID
from mantarray_file_manager import UTC_FIRST_TISSUE_DATA_POINT_UUID
from mantarray_file_manager import UTC_TIMESTAMP_OF_FILE_VERSION_MIGRATION_UUID
from mantarray_file_manager import WELL_COLUMN_UUID
from mantarray_file_manager import WELL_INDEX_UUID
from mantarray_file_manager import WELL_NAME_UUID
from mantarray_file_manager import WELL_ROW_UUID
from mantarray_file_manager import XEM_SERIAL_NUMBER_UUID


def test_default_UUIDs():
    assert CURI_BIO_ACCOUNT_UUID == uuid.UUID("73f52be0-368c-42d8-a1fd-660d49ba5604")
    assert CURI_BIO_USER_ACCOUNT_ID == uuid.UUID("455b93eb-c78f-4494-9f73-d3291130f126")


def test_time_conversion():
    assert DATETIME_STR_FORMAT == "%Y-%m-%d %H:%M:%S.%f"
    assert CENTIMILLISECONDS_PER_SECOND == 1e5
    assert MICROSECONDS_PER_CENTIMILLISECOND == 10


def test_versions():
    assert MIN_SUPPORTED_FILE_VERSION == "0.1.1"
    assert CURRENT_BETA1_HDF5_FILE_FORMAT_VERSION == "0.4.2"
    assert CURRENT_BETA2_HDF5_FILE_FORMAT_VERSION == "1.0.0"
    assert FILE_FORMAT_VERSION_METADATA_KEY == "File Format Version"


def test_metadata_UUIDs():
    assert NOT_APPLICABLE_H5_METADATA == uuid.UUID("59d92e00-99d5-4460-9a28-5a1a0fe9aecf")

    assert HARDWARE_TEST_RECORDING_UUID == uuid.UUID("a2e76058-08cd-475d-a55d-31d401c3cb34")
    assert UTC_BEGINNING_DATA_ACQUISTION_UUID == uuid.UUID("98c67f22-013b-421a-831b-0ea55df4651e")
    assert START_RECORDING_TIME_INDEX_UUID == uuid.UUID("e41422b3-c903-48fd-9856-46ff56a6534c")
    assert UTC_BEGINNING_RECORDING_UUID == uuid.UUID("d2449271-0e84-4b45-a28b-8deab390b7c2")
    assert UTC_FIRST_TISSUE_DATA_POINT_UUID == uuid.UUID("b32fb8cb-ebf8-4378-a2c0-f53a27bc77cc")
    assert UTC_FIRST_REF_DATA_POINT_UUID == uuid.UUID("7cc07b2b-4146-4374-b8f3-1c4d40ff0cf7")
    assert CUSTOMER_ACCOUNT_ID_UUID == uuid.UUID("4927c810-fbf4-406f-a848-eba5308576e6")
    assert USER_ACCOUNT_ID_UUID == uuid.UUID("7282cf00-2b6e-4202-9d9e-db0c73c3a71f")
    assert SOFTWARE_BUILD_NUMBER_UUID == uuid.UUID("b4db8436-10a4-4359-932d-aa80e6de5c76")
    assert SOFTWARE_RELEASE_VERSION_UUID == uuid.UUID("432fc3c1-051b-4604-bc3d-cc0d0bd75368")
    assert MAIN_FIRMWARE_VERSION_UUID == uuid.UUID("faa48a0c-0155-4234-afbf-5e5dbaa59537")
    assert SLEEP_FIRMWARE_VERSION_UUID == uuid.UUID("3a816076-90e4-4437-9929-dc910724a49d")
    assert XEM_SERIAL_NUMBER_UUID == uuid.UUID("e5f5b134-60c7-4881-a531-33aa0edba540")
    assert MANTARRAY_NICKNAME_UUID == uuid.UUID("0cdec9bb-d2b4-4c5b-9dd5-6a49766c5ed4")
    assert MANTARRAY_SERIAL_NUMBER_UUID == uuid.UUID("83720d36-b941-4d85-9b39-1d817799edd6")
    assert REFERENCE_VOLTAGE_UUID == uuid.UUID("0b3f3f56-0cc7-45f0-b748-9b9de480cba8")
    assert WELL_NAME_UUID == uuid.UUID("6d78f3b9-135a-4195-b014-e74dee70387b")
    assert WELL_ROW_UUID == uuid.UUID("da82fe73-16dd-456a-ac05-0b70fb7e0161")
    assert WELL_COLUMN_UUID == uuid.UUID("7af25a0a-8253-4d32-98c4-3c2ca0d83906")
    assert WELL_INDEX_UUID == uuid.UUID("cd89f639-1e36-4a13-a5ed-7fec6205f779")
    assert TOTAL_WELL_COUNT_UUID == uuid.UUID("7ca73e1c-9555-4eca-8281-3f844b5606dc")
    assert ADC_GAIN_SETTING_UUID == uuid.UUID("a3c3bb32-9b92-4da1-8ed8-6c09f9c816f8")
    assert ADC_TISSUE_OFFSET_UUID == uuid.UUID("41069860-159f-49f2-a59d-401783c1ecb4")
    assert ADC_REF_OFFSET_UUID == uuid.UUID("dc10066c-abf2-42b6-9b94-5e52d1ea9bfc")
    assert REF_SAMPLING_PERIOD_UUID == uuid.UUID("48aa034d-8775-453f-b135-75a983d6b553")
    assert TISSUE_SAMPLING_PERIOD_UUID == uuid.UUID("f629083a-3724-4100-8ece-c03e637ac19c")
    assert PLATE_BARCODE_UUID == uuid.UUID("cf60afef-a9f0-4bc3-89e9-c665c6bb6941")
    assert BACKEND_LOG_UUID == uuid.UUID("87533deb-2495-4430-bce7-12fdfc99158e")
    assert COMPUTER_NAME_HASH_UUID == uuid.UUID("fefd0675-35c2-45f6-855a-9500ad3f100d")
    assert BARCODE_IS_FROM_SCANNER_UUID == uuid.UUID("7d026e86-da70-4464-9181-dc0ce2d47bd1")
    assert IS_FILE_ORIGINAL_UNTRIMMED_UUID == uuid.UUID("52231a24-97a3-497a-917c-86c780d9993f")
    assert TRIMMED_TIME_FROM_ORIGINAL_START_UUID == uuid.UUID("371996e6-5e2d-4183-a5cf-06de7058210a")
    assert TRIMMED_TIME_FROM_ORIGINAL_END_UUID == uuid.UUID("55f6770d-c369-42ce-a437-5ed89c3cb1f8")
    assert ORIGINAL_FILE_VERSION_UUID == uuid.UUID("cd1b4063-4a87-4a57-bc12-923ff4890844")
    assert UTC_TIMESTAMP_OF_FILE_VERSION_MIGRATION_UUID == uuid.UUID("399b2148-09d4-418b-a132-e37df2721938")
    assert FILE_VERSION_PRIOR_TO_MIGRATION_UUID == uuid.UUID("11b4945b-3cf3-4f67-8bee-7abc3c449756")
    assert BOOTUP_COUNTER_UUID == uuid.UUID("b9ccc724-a39d-429a-be6d-3fd29be5037d")
    assert TOTAL_WORKING_HOURS_UUID == uuid.UUID("f8108718-2fa0-40ce-a51a-8478e5edd4b8")
    assert TAMPER_FLAG_UUID == uuid.UUID("68d0147f-9a84-4423-9c50-228da16ba895")
    assert PCB_SERIAL_NUMBER_UUID == uuid.UUID("5103f995-19d2-4880-8a2e-2ce9080cd2f5")
    assert MAGNETOMETER_CONFIGURATION_UUID == uuid.UUID("921121e9-4191-4536-bedd-03186fa1e117")
    assert UTC_BEGINNING_STIMULATION_UUID == uuid.UUID("4b310594-ded4-45fd-a1b4-b829aceb416c")
    assert STIMULATION_PROTOCOL_UUID == uuid.UUID("ede638ce-544e-427a-b1d9-c40784d7c82d")

    assert METADATA_UUID_DESCRIPTIONS == {
        HARDWARE_TEST_RECORDING_UUID: "Is Hardware Test Recording",
        UTC_BEGINNING_DATA_ACQUISTION_UUID: "UTC Timestamp of Beginning of Data Acquisition",
        UTC_BEGINNING_RECORDING_UUID: "UTC Timestamp of Beginning of Recording",
        UTC_FIRST_TISSUE_DATA_POINT_UUID: "UTC Timestamp of Beginning of Recorded Tissue Sensor Data",
        UTC_FIRST_REF_DATA_POINT_UUID: "UTC Timestamp of Beginning of Recorded Reference Sensor Data",
        START_RECORDING_TIME_INDEX_UUID: "Timepoint of Beginning of Recording",
        CUSTOMER_ACCOUNT_ID_UUID: "Customer Account ID",
        USER_ACCOUNT_ID_UUID: "User Account ID",
        SOFTWARE_BUILD_NUMBER_UUID: "Software Build Number",
        SOFTWARE_RELEASE_VERSION_UUID: "Software Release Version",
        MAIN_FIRMWARE_VERSION_UUID: "Firmware Version (Main Controller)",
        SLEEP_FIRMWARE_VERSION_UUID: "Firmware Version (Sleep Mode)",
        XEM_SERIAL_NUMBER_UUID: "XEM Serial Number",
        MANTARRAY_NICKNAME_UUID: "Mantarray Nickname",
        MANTARRAY_SERIAL_NUMBER_UUID: "Mantarray Serial Number",
        REFERENCE_VOLTAGE_UUID: "Reference Voltage",
        WELL_NAME_UUID: "Well Name",
        WELL_ROW_UUID: "Well Row (zero-based)",
        WELL_COLUMN_UUID: "Well Column (zero-based)",
        WELL_INDEX_UUID: "Well Index (zero-based)",
        TOTAL_WELL_COUNT_UUID: "Total Wells in Plate",
        REF_SAMPLING_PERIOD_UUID: "Reference Sensor Sampling Period (microseconds)",
        TISSUE_SAMPLING_PERIOD_UUID: "Tissue Sensor Sampling Period (microseconds)",
        ADC_GAIN_SETTING_UUID: "ADC Gain Setting",
        ADC_TISSUE_OFFSET_UUID: "ADC Tissue Sensor Offset",
        ADC_REF_OFFSET_UUID: "ADC Reference Sensor Offset",
        PLATE_BARCODE_UUID: "Plate Barcode",
        BACKEND_LOG_UUID: "Backend log file identifier",
        COMPUTER_NAME_HASH_UUID: "SHA512 digest of computer name",
        BARCODE_IS_FROM_SCANNER_UUID: "Is this barcode obtained from the scanner",
        IS_FILE_ORIGINAL_UNTRIMMED_UUID: "Is this an original file straight from the instrument and untrimmed",
        TRIMMED_TIME_FROM_ORIGINAL_START_UUID: "Number of centimilliseconds if Beta 1 data or microseconds o/w that has been trimmed off the beginning of when the original data started",
        TRIMMED_TIME_FROM_ORIGINAL_END_UUID: "Number of centimilliseconds if Beta 1 data or microseconds o/w that has been trimmed off the end of when the original data ended",
        ORIGINAL_FILE_VERSION_UUID: "The original version of the file when recorded, prior to any migrations to newer versions/formats.",
        UTC_TIMESTAMP_OF_FILE_VERSION_MIGRATION_UUID: "Timestamp when this file was migrated from an earlier version.",
        FILE_VERSION_PRIOR_TO_MIGRATION_UUID: "File format version that this file was migrated from",
        BOOTUP_COUNTER_UUID: "The number of times this Mantarray Instrument has booted up",
        TOTAL_WORKING_HOURS_UUID: "The total number of hours this Mantarray Instrument has been powered on and running",
        TAMPER_FLAG_UUID: "Is it suspected the internals of the Mantarray enclosure have been tampered with",
        PCB_SERIAL_NUMBER_UUID: "The serial number of the Mantarray PCB",
        MAGNETOMETER_CONFIGURATION_UUID: "The board's magnetometer channels that were enabled during this recording",
        UTC_BEGINNING_STIMULATION_UUID: "UTC Timestamp of Beginning of Stimulation",
        STIMULATION_PROTOCOL_UUID: "The stimulation protocol that was running on this well during recording. Empty string if stimulation was not active",
    }


def test_file_migration_paths():
    assert FILE_MIGRATION_PATHS == {"0.3.1": "0.4.1", "0.4.1": "0.4.2"}


def test_sensor_data_types():
    assert TISSUE_SENSOR_READINGS == "tissue_sensor_readings"
    assert REFERENCE_SENSOR_READINGS == "reference_sensor_readings"
    assert STIMULATION_READINGS == "stimulation_readings"
    assert TIME_INDICES == "time_indices"
    assert TIME_OFFSETS == "time_offsets"
