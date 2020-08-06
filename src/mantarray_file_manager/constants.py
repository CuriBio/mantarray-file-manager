# -*- coding: utf-8 -*-
"""Constants for the Mantarray File Manager."""
import uuid

CURI_BIO_ACCOUNT_UUID = uuid.UUID("73f52be0-368c-42d8-a1fd-660d49ba5604")
CURI_BIO_USER_ACCOUNT_ID = uuid.UUID("455b93eb-c78f-4494-9f73-d3291130f126")
DEFAULT_USER_CONFIG = {
    "Customer Account ID": "",
    "User Account ID": "",
}

HARDWARE_TEST_RECORDING_UUID = uuid.UUID("a2e76058-08cd-475d-a55d-31d401c3cb34")
UTC_BEGINNING_DATA_ACQUISTION_UUID = uuid.UUID("98c67f22-013b-421a-831b-0ea55df4651e")
START_RECORDING_TIME_INDEX_UUID = uuid.UUID("e41422b3-c903-48fd-9856-46ff56a6534c")
UTC_BEGINNING_RECORDING_UUID = uuid.UUID("d2449271-0e84-4b45-a28b-8deab390b7c2")
UTC_FIRST_TISSUE_DATA_POINT_UUID = uuid.UUID("b32fb8cb-ebf8-4378-a2c0-f53a27bc77cc")
UTC_FIRST_REF_DATA_POINT_UUID = uuid.UUID("7cc07b2b-4146-4374-b8f3-1c4d40ff0cf7")
CUSTOMER_ACCOUNT_ID_UUID = uuid.UUID("4927c810-fbf4-406f-a848-eba5308576e6")
USER_ACCOUNT_ID_UUID = uuid.UUID("7282cf00-2b6e-4202-9d9e-db0c73c3a71f")
SOFTWARE_BUILD_NUMBER_UUID = uuid.UUID("b4db8436-10a4-4359-932d-aa80e6de5c76")
SOFTWARE_RELEASE_VERSION_UUID = uuid.UUID("432fc3c1-051b-4604-bc3d-cc0d0bd75368")
MAIN_FIRMWARE_VERSION_UUID = uuid.UUID("faa48a0c-0155-4234-afbf-5e5dbaa59537")
SLEEP_FIRMWARE_VERSION_UUID = uuid.UUID("3a816076-90e4-4437-9929-dc910724a49d")
XEM_SERIAL_NUMBER_UUID = uuid.UUID("e5f5b134-60c7-4881-a531-33aa0edba540")
MANTARRAY_NICKNAME_UUID = uuid.UUID("0cdec9bb-d2b4-4c5b-9dd5-6a49766c5ed4")
MANTARRAY_SERIAL_NUMBER_UUID = uuid.UUID("83720d36-b941-4d85-9b39-1d817799edd6")
REFERENCE_VOLTAGE_UUID = uuid.UUID("0b3f3f56-0cc7-45f0-b748-9b9de480cba8")
WELL_NAME_UUID = uuid.UUID("6d78f3b9-135a-4195-b014-e74dee70387b")
WELL_ROW_UUID = uuid.UUID("da82fe73-16dd-456a-ac05-0b70fb7e0161")
WELL_COLUMN_UUID = uuid.UUID("7af25a0a-8253-4d32-98c4-3c2ca0d83906")
WELL_INDEX_UUID = uuid.UUID("cd89f639-1e36-4a13-a5ed-7fec6205f779")
TOTAL_WELL_COUNT_UUID = uuid.UUID("7ca73e1c-9555-4eca-8281-3f844b5606dc")
REF_SAMPLING_PERIOD_UUID = uuid.UUID("48aa034d-8775-453f-b135-75a983d6b553")
TISSUE_SAMPLING_PERIOD_UUID = uuid.UUID("f629083a-3724-4100-8ece-c03e637ac19c")
ADC_GAIN_SETTING_UUID = uuid.UUID("a3c3bb32-9b92-4da1-8ed8-6c09f9c816f8")
ADC_TISSUE_OFFSET_UUID = uuid.UUID("41069860-159f-49f2-a59d-401783c1ecb4")
ADC_REF_OFFSET_UUID = uuid.UUID("dc10066c-abf2-42b6-9b94-5e52d1ea9bfc")
PLATE_BARCODE_UUID = uuid.UUID("cf60afef-a9f0-4bc3-89e9-c665c6bb6941")
METADATA_UUID_DESCRIPTIONS = {
    HARDWARE_TEST_RECORDING_UUID: "Is Hardware Test Recording",
    START_RECORDING_TIME_INDEX_UUID: "Timepoint of Beginning of Recording",
    UTC_BEGINNING_DATA_ACQUISTION_UUID: "UTC Timestamp of Beginning of Data Acquisition",
    UTC_BEGINNING_RECORDING_UUID: "UTC Timestamp of Beginning of Recording",
    UTC_FIRST_TISSUE_DATA_POINT_UUID: "UTC Timestamp of Beginning of Recorded Tissue Sensor Data",
    UTC_FIRST_REF_DATA_POINT_UUID: "UTC Timestamp of Beginning of Recorded Reference Sensor Data",
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
}