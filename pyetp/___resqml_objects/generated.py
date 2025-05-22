from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Any
from xsdata.models.datatype import XmlDate, XmlDateTime, XmlPeriod

import sys
py310 = sys.version_info.minor >= 10 or sys.version_info.major > 3

class APIGammaRayUom(Enum):
    """
    Properties
    ----------
    G_API
        API gamma ray unit
    """

    G_API = "gAPI"


class APIGravityUom(Enum):
    """
    Properties
    ----------
    D_API
        API gravity unit
    """

    D_API = "dAPI"


class APINeutronUom(Enum):
    """
    Properties
    ----------
    N_API
        API neutron unit
    """

    N_API = "nAPI"


class AbsorbedDoseUom(Enum):
    """
    Properties
    ----------
    C_GY
        centigray
    CRD
        hundredth of rad
    D_GY
        decigray
    DRD
        tenth of rad
    EGY
        exagray
    ERD
        million million million rad
    F_GY
        femtogray
    FRD
        femtorad
    GGY
        gigagray
    GRD
        thousand million rad
    GY
        gray
    K_GY
        kilogray
    KRD
        thousand rad
    M_GY
        milligray
    MGY_1
        megagray
    MRD
        million rad
    MRD_1
        thousandth of rad
    N_GY
        nanogray
    NRD
        nanorad
    P_GY
        picogray
    PRD
        picorad
    RD
        rad
    TGY
        teragray
    TRD
        million million rad
    U_GY
        microgray
    URD
        millionth of rad
    """

    C_GY = "cGy"
    CRD = "crd"
    D_GY = "dGy"
    DRD = "drd"
    EGY = "EGy"
    ERD = "Erd"
    F_GY = "fGy"
    FRD = "frd"
    GGY = "GGy"
    GRD = "Grd"
    GY = "Gy"
    K_GY = "kGy"
    KRD = "krd"
    M_GY = "mGy"
    MGY_1 = "MGy"
    MRD = "Mrd"
    MRD_1 = "mrd"
    N_GY = "nGy"
    NRD = "nrd"
    P_GY = "pGy"
    PRD = "prd"
    RD = "rd"
    TGY = "TGy"
    TRD = "Trd"
    U_GY = "uGy"
    URD = "urd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractProjectedCrs:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractVerticalCrs:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"


class ActivityOfRadioactivityUom(Enum):
    """
    Properties
    ----------
    BQ
        becquerel
    CI
        curie
    GBQ
        gigabecquerel
    MBQ
        megabecquerel
    M_CI
        thousandth of curie
    N_CI
        nanocurie
    P_CI
        picocurie
    TBQ
        terabecquerel
    U_CI
        millionth of curie
    """

    BQ = "Bq"
    CI = "Ci"
    GBQ = "GBq"
    MBQ = "MBq"
    M_CI = "mCi"
    N_CI = "nCi"
    P_CI = "pCi"
    TBQ = "TBq"
    U_CI = "uCi"


class AmountOfSubstancePerAmountOfSubstanceUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    MOLAR
        percent [molar basis]
    EUC
        euclid
    MOL_MOL
        mole per mole
    N_EUC
        nanoeuclid
    PPK
        part per thousand
    PPM
        part per million
    """

    VALUE = "%"
    MOLAR = "%[molar]"
    EUC = "Euc"
    MOL_MOL = "mol/mol"
    N_EUC = "nEuc"
    PPK = "ppk"
    PPM = "ppm"


class AmountOfSubstancePerAreaUom(Enum):
    """
    Properties
    ----------
    MOL_M2
        gram-mole per square metre
    """

    MOL_M2 = "mol/m2"


class AmountOfSubstancePerTimePerAreaUom(Enum):
    """
    Properties
    ----------
    LBMOL_H_FT2
        pound-mass-mole per hour square foot
    LBMOL_S_FT2
        pound-mass-mole per second square foot
    MOL_S_M2
        gram-mole per second square metre
    """

    LBMOL_H_FT2 = "lbmol/(h.ft2)"
    LBMOL_S_FT2 = "lbmol/(s.ft2)"
    MOL_S_M2 = "mol/(s.m2)"


class AmountOfSubstancePerTimeUom(Enum):
    """
    Properties
    ----------
    KMOL_H
        kilogram-mole per hour
    KMOL_S
        kilogram-mole per second
    LBMOL_H
        pound-mass-mole per hour
    LBMOL_S
        pound-mass-mole per second
    MOL_S
        gram-mole per second
    """

    KMOL_H = "kmol/h"
    KMOL_S = "kmol/s"
    LBMOL_H = "lbmol/h"
    LBMOL_S = "lbmol/s"
    MOL_S = "mol/s"


class AmountOfSubstancePerVolumeUom(Enum):
    """
    Properties
    ----------
    KMOL_M3
        kilogram-mole per cubic metre
    LBMOL_FT3
        pound-mass-mole per cubic foot
    LBMOL_GAL_UK
        pound-mass-mole per UK gallon
    LBMOL_GAL_US
        pound-mass-mole per US gallon
    MOL_M3
        gram-mole per cubic metre
    """

    KMOL_M3 = "kmol/m3"
    LBMOL_FT3 = "lbmol/ft3"
    LBMOL_GAL_UK = "lbmol/gal[UK]"
    LBMOL_GAL_US = "lbmol/gal[US]"
    MOL_M3 = "mol/m3"


class AmountOfSubstanceUom(Enum):
    """
    Properties
    ----------
    KMOL
        kilogram-mole
    LBMOL
        pound-mass-mole
    MMOL
        milligram-mole
    MOL
        gram-mole
    UMOL
        microgram-mole
    """

    KMOL = "kmol"
    LBMOL = "lbmol"
    MMOL = "mmol"
    MOL = "mol"
    UMOL = "umol"


class AnglePerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_DEGA_FT
        angular degree per hundred foot
    VALUE_1_30_DEGA_FT
        angular degree per thirty foot
    VALUE_1_30_DEGA_M
        angular degree per thirty metre
    DEGA_FT
        angular degree per foot
    DEGA_M
        angular degree per metre
    RAD_FT
        radian per foot
    RAD_M
        radian per metre
    REV_FT
        revolution per foot
    REV_M
        revolution per metre
    """

    VALUE_0_01_DEGA_FT = "0.01 dega/ft"
    VALUE_1_30_DEGA_FT = "1/30 dega/ft"
    VALUE_1_30_DEGA_M = "1/30 dega/m"
    DEGA_FT = "dega/ft"
    DEGA_M = "dega/m"
    RAD_FT = "rad/ft"
    RAD_M = "rad/m"
    REV_FT = "rev/ft"
    REV_M = "rev/m"


class AnglePerVolumeUom(Enum):
    """
    Properties
    ----------
    RAD_FT3
        radian per cubic foot
    RAD_M3
        radian per cubic metre
    """

    RAD_FT3 = "rad/ft3"
    RAD_M3 = "rad/m3"


class AngularAccelerationUom(Enum):
    """
    Properties
    ----------
    RAD_S2
        radian per second squared
    RPM_S
        (revolution per minute) per second
    """

    RAD_S2 = "rad/s2"
    RPM_S = "rpm/s"


class AngularVelocityUom(Enum):
    """
    Properties
    ----------
    DEGA_H
        angular degree per hour
    DEGA_MIN
        angular degree per minute
    DEGA_S
        angular degree per second
    RAD_S
        radian per second
    REV_S
        revolution per second
    RPM
        revolution per minute
    """

    DEGA_H = "dega/h"
    DEGA_MIN = "dega/min"
    DEGA_S = "dega/s"
    RAD_S = "rad/s"
    REV_S = "rev/s"
    RPM = "rpm"


class AreaPerAmountOfSubstanceUom(Enum):
    """
    Properties
    ----------
    M2_MOL
        square metre per gram-mole
    """

    M2_MOL = "m2/mol"


class AreaPerAreaUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    AREA
        percent [area basis]
    C_EUC
        centieuclid
    EUC
        euclid
    IN2_FT2
        square inch per square foot
    IN2_IN2
        square inch per square inch
    M2_M2
        square metre per square metre
    MM2_MM2
        square millimetre per square millimetre
    """

    VALUE = "%"
    AREA = "%[area]"
    C_EUC = "cEuc"
    EUC = "Euc"
    IN2_FT2 = "in2/ft2"
    IN2_IN2 = "in2/in2"
    M2_M2 = "m2/m2"
    MM2_MM2 = "mm2/mm2"


class AreaPerMassUom(Enum):
    """
    Properties
    ----------
    CM2_G
        square centimetre per gram
    FT2_LBM
        square foot per pound-mass
    M2_G
        square metre per gram
    M2_KG
        square metre per kilogram
    """

    CM2_G = "cm2/g"
    FT2_LBM = "ft2/lbm"
    M2_G = "m2/g"
    M2_KG = "m2/kg"


class AreaPerTimeUom(Enum):
    """
    Properties
    ----------
    CM2_S
        square centimetre per second
    FT2_H
        square foot per hour
    FT2_S
        square foot per second
    IN2_S
        square inch per second
    M2_D
        square metre per day
    M2_H
        square metre per hour
    M2_S
        square metre per second
    MM2_S
        square millimetre per second
    """

    CM2_S = "cm2/s"
    FT2_H = "ft2/h"
    FT2_S = "ft2/s"
    IN2_S = "in2/s"
    M2_D = "m2/d"
    M2_H = "m2/h"
    M2_S = "m2/s"
    MM2_S = "mm2/s"


class AreaPerVolumeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_M
        per metre
    B_CM3
        barn per cubic centimetre
    CU
        capture unit
    FT2_IN3
        square foot per cubic inch
    M2_CM3
        square metre per cubic centimetre
    M2_M3
        square metre per cubic metre
    """

    VALUE_1_M = "1/m"
    B_CM3 = "b/cm3"
    CU = "cu"
    FT2_IN3 = "ft2/in3"
    M2_CM3 = "m2/cm3"
    M2_M3 = "m2/m3"


class AreaUom(Enum):
    """
    Properties
    ----------
    ACRE
        acre
    B
        barn
    CM2
        square centimetre
    FT2
        square foot
    HA
        hectare
    IN2
        square inch
    KM2
        square kilometre
    M2
        square metre
    MI_US_2
        square US survey mile
    MI2
        square mile
    MM2
        square millimetre
    SECTION
        section
    UM2
        square micrometre
    YD2
        square yard
    """

    ACRE = "acre"
    B = "b"
    CM2 = "cm2"
    FT2 = "ft2"
    HA = "ha"
    IN2 = "in2"
    KM2 = "km2"
    M2 = "m2"
    MI_US_2 = "mi[US]2"
    MI2 = "mi2"
    MM2 = "mm2"
    SECTION = "section"
    UM2 = "um2"
    YD2 = "yd2"


class AttenuationPerFrequencyIntervalUom(Enum):
    """
    Properties
    ----------
    B_O
        bel per octave
    D_B_O
        decibel per octave
    """

    B_O = "B/O"
    D_B_O = "dB/O"


class AxisOrder2d(Enum):
    """
    Defines the cordinate system axis order of the global CRS using the axis names
    (from EPSG database).

    Properties
    ----------
    EASTING_NORTHING
        The first axis is easting and the second axis is northing.
    NORTHING_EASTING
        The first axis is northing and the second asis is easting.
    WESTING_SOUTHING
        The first axis is westing and the second axis is southing.
    SOUTHING_WESTING
        The first axis is southing and the second axis is westing.
    NORTHING_WESTING
        the first axis is northing and the second axis is westing.
    WESTING_NORTHING
        the first axis is westing and the second axis is northing.
    """

    EASTING_NORTHING = "easting northing"
    NORTHING_EASTING = "northing easting"
    WESTING_SOUTHING = "westing southing"
    SOUTHING_WESTING = "southing westing"
    NORTHING_WESTING = "northing westing"
    WESTING_NORTHING = "westing northing"


class CapacitanceUom(Enum):
    """
    Properties
    ----------
    C_F
        centifarad
    D_F
        decifarad
    EF
        exafarad
    F
        farad
    F_F
        femtofarad
    GF
        gigafarad
    K_F
        kilofarad
    M_F
        millifarad
    MF_1
        megafarad
    N_F
        nanofarad
    P_F
        picofarad
    TF
        terafarad
    U_F
        microfarad
    """

    C_F = "cF"
    D_F = "dF"
    EF = "EF"
    F = "F"
    F_F = "fF"
    GF = "GF"
    K_F = "kF"
    M_F = "mF"
    MF_1 = "MF"
    N_F = "nF"
    P_F = "pF"
    TF = "TF"
    U_F = "uF"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Citation:
    """
    An ISO 19115 EIP-derived set of metadata attached to all specializations of
    AbstractObject to ensure the traceability of each individual independent (top
    level) element.

    Parameters
    ----------
    title
        One line description/name of the RESQML object. This is the
        equivalent in ISO 19115 of CI_Citation.title Legacy DCGroup - title
    originator
        Name (or other human-readable identifier) of the person who
        initially originated the object or RESQML document in the source
        application. If that information is not available, the user who
        created the RESQML format file. The originator remains the same as
        the object is subsequently edited. This is the equivalent in ISO
        19115 to the CI_Individual.name or the CI_Organization.name of the
        citedResponsibleParty whose role is "originator". Legacy DCGroup -
        author
    creation
        Date and time the document was created in the source application or,
        if that information is not available, when it was saved to the
        RESQML format file. This is the equivalent of the ISO 19115 CI_Date
        where the CI_DateTypeCode = ”creation" The type is the Energistics
        timestamp datatype which is the W3C xs:dateTime with the optional
        timezone offset from UTC made mandatory. Format: YYYY-MM-
        DDThh:mm:ssZ[+/-]hh:mm Legacy DCGroup - created
    format
        Software or service that was used to originate the object and the
        file format created. Must be human and machine readable and
        unambiguously identify the software by including the company name,
        software name and software version. This is the equivalent in ISO
        19115 to the distributionFormat.MD_Format. The ISO format for this
        is [vendor:applicationName]/fileExtension where the application name
        includes the version number of the application. SIG Implementation
        Notes 1. RESQML - Legacy DCGroup from v1.1 - publisher -
        fileExtension is not relevant and will be ignored if present. -
        vendor and applicationName are mandatory.
    editor
        Name (or other human-readable identifier) of the last person who
        updated the object. This is the equivalent in ISO 19115 to the
        CI_Individual.name or the CI_Organization.name of the
        citedResponsibleParty whose role is "editor". Legacy DCGroup -
        contributor
    last_update
        Date and time the document was last modified in the source
        application or, if that information is not available, when it was
        last saved to the RESQML format file. This is the equivalent of the
        ISO 19115 CI_Date where the CI_DateTypeCode = ”lastUpdate" The type
        is the Energistics timestamp datatype which is the W3C xs:dateTime
        with the optional timezone offset from UTC made mandatory. Format:
        YYYY-MM-DDThh:mm:ssZ[+/-]hh:mm Legacy DCGroup - modified
    version_string
    description
        User descriptive comments about the object. Intended for end-user
        use (human readable); not necessarily meant to be used by software.
        This is the equivalent of the ISO 19115 abstract.CharacterString
        Legacy DCGroup - description
    descriptive_keywords
        Key words to describe the activity, for example, history match or
        volumetric calculations, relevant to this object. Intended to be
        used in a search function by software. This is the equivalent in ISO
        19115 of descriptiveKeywords.MD_Keywords Legacy DCGroup - subject
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    title: str = field(
        metadata={
            "name": "Title",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "min_length": 1,
            "max_length": 256,
            "white_space": "collapse",
        }
    )
    originator: str = field(
        metadata={
            "name": "Originator",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "min_length": 1,
            "max_length": 64,
            "white_space": "collapse",
        }
    )
    creation: XmlDateTime = field(
        metadata={
            "name": "Creation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    format: str = field(
        metadata={
            "name": "Format",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "min_length": 1,
            "max_length": 256,
            "white_space": "collapse",
        }
    )
    editor: Optional[str] = field(
        default=None,
        metadata={
            "name": "Editor",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "min_length": 1,
            "max_length": 64,
            "white_space": "collapse",
        },
    )
    last_update: Optional[XmlDateTime] = field(
        default=None,
        metadata={
            "name": "LastUpdate",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
        },
    )
    version_string: Optional[str] = field(
        default=None,
        metadata={
            "name": "VersionString",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
        },
    )
    description: Optional[str] = field(
        default=None,
        metadata={
            "name": "Description",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "min_length": 1,
            "max_length": 4000,
            "white_space": "collapse",
        },
    )
    descriptive_keywords: Optional[str] = field(
        default=None,
        metadata={
            "name": "DescriptiveKeywords",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "min_length": 1,
            "max_length": 4000,
            "white_space": "collapse",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CustomData:
    """WITSML - Custom or User Defined Element and Attributes Component Schema.
    Specify custom element, attributes, and types in the custom data area.

    Parameters
    ----------
    any_element
        Any element or attribute in any namespace. It is strongly
        recommended that all custom data definitions be added to a unique
        namespace.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    any_element: List[object] = field(
        default_factory=list,
        metadata={
            "type": "Wildcard",
            "namespace": "##any",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DataObjectReference:
    """
    It only applies for Energistics data object.

    Parameters
    ----------
    content_type
        The content type of the referenced element.
    title
    uuid
        Reference to an object using its global UID.
    uuid_authority
        The authority that issued and maintains the uuid of the referenced
        object. Used mainly in alias context.
    version_string
        Indicates the version of the object which is referenced.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    content_type: str = field(
        metadata={
            "name": "ContentType",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    title: str = field(
        metadata={
            "name": "Title",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "min_length": 1,
            "max_length": 256,
            "white_space": "collapse",
        }
    )
    uuid: str = field(
        metadata={
            "name": "UUID",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "pattern": r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
        }
    )
    uuid_authority: Optional[str] = field(
        default=None,
        metadata={
            "name": "UuidAuthority",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
        },
    )
    version_string: Optional[str] = field(
        default=None,
        metadata={
            "name": "VersionString",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "min_length": 1,
            "max_length": 64,
            "white_space": "collapse",
        },
    )


class DataTransferSpeedUom(Enum):
    """
    Properties
    ----------
    BIT_S
        bit per second
    BYTE_S
        byte per second
    """

    BIT_S = "bit/s"
    BYTE_S = "byte/s"


class DiffusionCoefficientUom(Enum):
    """
    Properties
    ----------
    M2_S
        square metre per second
    """

    M2_S = "m2/s"


class DigitalStorageUom(Enum):
    """
    Properties
    ----------
    BIT
        bit
    BYTE
        byte
    KIBYTE
        kibibyte
    MIBYTE
        mebibyte
    """

    BIT = "bit"
    BYTE = "byte"
    KIBYTE = "Kibyte"
    MIBYTE = "Mibyte"


class DimensionlessUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    C_EUC
        centieuclid
    D_EUC
        decieuclid
    EEUC
        exaeuclid
    EUC
        euclid
    F_EUC
        femtoeuclid
    GEUC
        gigaeuclid
    K_EUC
        kiloeuclid
    MEUC
        megaeuclid
    M_EUC_1
        millieuclid
    N_EUC
        nanoeuclid
    P_EUC
        picoeuclid
    PPK
        part per thousand
    PPM
        part per million
    TEUC
        teraeuclid
    U_EUC
        microeuclid
    """

    VALUE = "%"
    C_EUC = "cEuc"
    D_EUC = "dEuc"
    EEUC = "EEuc"
    EUC = "Euc"
    F_EUC = "fEuc"
    GEUC = "GEuc"
    K_EUC = "kEuc"
    MEUC = "MEuc"
    M_EUC_1 = "mEuc"
    N_EUC = "nEuc"
    P_EUC = "pEuc"
    PPK = "ppk"
    PPM = "ppm"
    TEUC = "TEuc"
    U_EUC = "uEuc"


class DipoleMomentUom(Enum):
    """
    Properties
    ----------
    C_M
        coulomb metre
    """

    C_M = "C.m"


class DoseEquivalentUom(Enum):
    """
    Properties
    ----------
    MREM
        thousandth of rem
    M_SV
        millisievert
    REM
        rem
    SV
        sievert
    """

    MREM = "mrem"
    M_SV = "mSv"
    REM = "rem"
    SV = "Sv"


class DynamicViscosityUom(Enum):
    """
    Properties
    ----------
    C_P
        centipoise
    D_P
        decipoise
    DYNE_S_CM2
        dyne second per square centimetre
    EP
        exapoise
    F_P
        femtopoise
    GP
        gigapoise
    KGF_S_M2
        thousand gram-force second per square metre
    K_P
        kilopoise
    LBF_S_FT2
        pound-force second per square foot
    LBF_S_IN2
        pound-force second per square inch
    M_P
        millipoise
    MP_1
        megapoise
    M_PA_S
        millipascal second
    N_S_M2
        newton second per square metre
    N_P
        nanopoise
    P
        poise
    PA_S
        pascal second
    P_P
        picopoise
    PSI_S
        psi second
    TP
        terapoise
    U_P
        micropoise
    """

    C_P = "cP"
    D_P = "dP"
    DYNE_S_CM2 = "dyne.s/cm2"
    EP = "EP"
    F_P = "fP"
    GP = "GP"
    KGF_S_M2 = "kgf.s/m2"
    K_P = "kP"
    LBF_S_FT2 = "lbf.s/ft2"
    LBF_S_IN2 = "lbf.s/in2"
    M_P = "mP"
    MP_1 = "MP"
    M_PA_S = "mPa.s"
    N_S_M2 = "N.s/m2"
    N_P = "nP"
    P = "P"
    PA_S = "Pa.s"
    P_P = "pP"
    PSI_S = "psi.s"
    TP = "TP"
    U_P = "uP"


class ElectricChargePerAreaUom(Enum):
    """
    Properties
    ----------
    C_CM2
        coulomb per square centimetre
    C_M2
        coulomb per square metre
    C_MM2
        coulomb per square millimetre
    M_C_M2
        millicoulomb per square metre
    """

    C_CM2 = "C/cm2"
    C_M2 = "C/m2"
    C_MM2 = "C/mm2"
    M_C_M2 = "mC/m2"


class ElectricChargePerMassUom(Enum):
    """
    Properties
    ----------
    A_S_KG
        ampere second per kilogram
    C_G
        coulomb per gram
    C_KG
        coulomb per kilogram
    """

    A_S_KG = "A.s/kg"
    C_G = "C/g"
    C_KG = "C/kg"


class ElectricChargePerVolumeUom(Enum):
    """
    Properties
    ----------
    A_S_M3
        ampere second per cubic metre
    C_CM3
        coulomb per cubic centimetre
    C_M3
        coulomb per cubic metre
    C_MM3
        coulomb per cubic millimetre
    """

    A_S_M3 = "A.s/m3"
    C_CM3 = "C/cm3"
    C_M3 = "C/m3"
    C_MM3 = "C/mm3"


class ElectricChargeUom(Enum):
    """
    Properties
    ----------
    A_H
        ampere hour
    A_S
        ampere second
    C
        coulomb
    C_C
        centicoulomb
    D_C
        decicoulomb
    EC
        exacoulomb
    F_C
        femtocoulomb
    GC
        gigacoulomb
    K_C
        kilocoulomb
    MC
        megacoulomb
    M_C_1
        millicoulomb
    N_C
        nanocoulomb
    P_C
        picocoulomb
    TC
        teracoulomb
    U_C
        microcoulomb
    """

    A_H = "A.h"
    A_S = "A.s"
    C = "C"
    C_C = "cC"
    D_C = "dC"
    EC = "EC"
    F_C = "fC"
    GC = "GC"
    K_C = "kC"
    MC = "MC"
    M_C_1 = "mC"
    N_C = "nC"
    P_C = "pC"
    TC = "TC"
    U_C = "uC"


class ElectricConductanceUom(Enum):
    """
    Properties
    ----------
    C_S
        centisiemens
    D_S
        decisiemens
    ES
        exasiemens
    F_S
        femtosiemens
    GS
        gigasiemens
    K_S
        kilosiemens
    M_S
        millisiemens
    MS_1
        megasiemens
    N_S
        nanosiemens
    P_S
        picosiemens
    S
        siemens
    TS
        terasiemens
    U_S
        microsiemens
    """

    C_S = "cS"
    D_S = "dS"
    ES = "ES"
    F_S = "fS"
    GS = "GS"
    K_S = "kS"
    M_S = "mS"
    MS_1 = "MS"
    N_S = "nS"
    P_S = "pS"
    S = "S"
    TS = "TS"
    U_S = "uS"


class ElectricConductivityUom(Enum):
    """
    Properties
    ----------
    K_S_M
        kilosiemens per metre
    M_S_CM
        millisiemens per centimetre
    M_S_M
        millisiemens per metre
    S_M
        siemens per metre
    """

    K_S_M = "kS/m"
    M_S_CM = "mS/cm"
    M_S_M = "mS/m"
    S_M = "S/m"


class ElectricCurrentDensityUom(Enum):
    """
    Properties
    ----------
    A_CM2
        ampere per square centimetre
    A_FT2
        ampere per square foot
    A_M2
        ampere per square metre
    A_MM2
        ampere per square millimetre
    M_A_CM2
        milliampere per square centimetre
    M_A_FT2
        milliampere per square foot
    U_A_CM2
        microampere per square centimetre
    U_A_IN2
        microampere per square inch
    """

    A_CM2 = "A/cm2"
    A_FT2 = "A/ft2"
    A_M2 = "A/m2"
    A_MM2 = "A/mm2"
    M_A_CM2 = "mA/cm2"
    M_A_FT2 = "mA/ft2"
    U_A_CM2 = "uA/cm2"
    U_A_IN2 = "uA/in2"


class ElectricCurrentUom(Enum):
    """
    Properties
    ----------
    A
        ampere
    C_A
        centiampere
    D_A
        deciampere
    EA
        exaampere
    F_A
        femtoampere
    GA
        gigaampere
    K_A
        kiloampere
    M_A
        milliampere
    MA_1
        megaampere
    N_A
        nanoampere
    P_A
        picoampere
    TA
        teraampere
    U_A
        microampere
    """

    A = "A"
    C_A = "cA"
    D_A = "dA"
    EA = "EA"
    F_A = "fA"
    GA = "GA"
    K_A = "kA"
    M_A = "mA"
    MA_1 = "MA"
    N_A = "nA"
    P_A = "pA"
    TA = "TA"
    U_A = "uA"


class ElectricFieldStrengthUom(Enum):
    """
    Properties
    ----------
    M_V_FT
        millivolt per foot
    M_V_M
        millivolt per metre
    U_V_FT
        microvolt per foot
    U_V_M
        microvolt per metre
    V_M
        volt per metre
    """

    M_V_FT = "mV/ft"
    M_V_M = "mV/m"
    U_V_FT = "uV/ft"
    U_V_M = "uV/m"
    V_M = "V/m"


class ElectricPotentialDifferenceUom(Enum):
    """
    Properties
    ----------
    C_V
        centivolt
    D_V
        decivolt
    F_V
        femtovolt
    GV
        gigavolt
    K_V
        kilovolt
    M_V
        millivolt
    MV_1
        megavolt
    N_V
        nanovolt
    P_V
        picovolt
    TV
        teravolt
    U_V
        microvolt
    V
        volt
    """

    C_V = "cV"
    D_V = "dV"
    F_V = "fV"
    GV = "GV"
    K_V = "kV"
    M_V = "mV"
    MV_1 = "MV"
    N_V = "nV"
    P_V = "pV"
    TV = "TV"
    U_V = "uV"
    V = "V"


class ElectricResistancePerLengthUom(Enum):
    """
    Properties
    ----------
    OHM_M
        ohm per metre
    UOHM_FT
        microhm per foot
    UOHM_M
        microhm per metre
    """

    OHM_M = "ohm/m"
    UOHM_FT = "uohm/ft"
    UOHM_M = "uohm/m"


class ElectricResistanceUom(Enum):
    """
    Properties
    ----------
    COHM
        centiohm
    DOHM
        deciohm
    EOHM
        exaohm
    FOHM
        femtoohm
    GOHM
        gigaohm
    KOHM
        kilohm
    MOHM
        megohm
    MOHM_1
        milliohm
    NOHM
        nanoohm
    OHM
        ohm
    POHM
        picoohm
    TOHM
        teraohm
    UOHM
        microohm
    """

    COHM = "cohm"
    DOHM = "dohm"
    EOHM = "Eohm"
    FOHM = "fohm"
    GOHM = "Gohm"
    KOHM = "kohm"
    MOHM = "Mohm"
    MOHM_1 = "mohm"
    NOHM = "nohm"
    OHM = "ohm"
    POHM = "pohm"
    TOHM = "Tohm"
    UOHM = "uohm"


class ElectricalResistivityUom(Enum):
    """
    Properties
    ----------
    KOHM_M
        kiloohm metre
    NOHM_MIL2_FT
        nanoohm square mil per foot
    NOHM_MM2_M
        nanoohm square milimetre per metre
    OHM_CM
        ohm centimetre
    OHM_M
        ohm metre
    OHM_M2_M
        ohm square metre per metre
    """

    KOHM_M = "kohm.m"
    NOHM_MIL2_FT = "nohm.mil2/ft"
    NOHM_MM2_M = "nohm.mm2/m"
    OHM_CM = "ohm.cm"
    OHM_M = "ohm.m"
    OHM_M2_M = "ohm.m2/m"


class ElectromagneticMomentUom(Enum):
    """
    Properties
    ----------
    A_M2
        ampere square metre
    """

    A_M2 = "A.m2"


class EnergyLengthPerAreaUom(Enum):
    """
    Properties
    ----------
    J_M_M2
        joule metre per square metre
    KCAL_TH_M_CM2
        thousand calorie metre per square centimetre
    """

    J_M_M2 = "J.m/m2"
    KCAL_TH_M_CM2 = "kcal[th].m/cm2"


class EnergyLengthPerTimeAreaTemperatureUom(Enum):
    """
    Properties
    ----------
    BTU_IT_IN_H_FT2_DELTA_F
        BTU per (hour square foot delta Fahrenheit per inch)
    J_M_S_M2_DELTA_K
        joule metre per second square metre delta kelvin
    K_J_M_H_M2_DELTA_K
        kilojoule metre per hour square metre delta kelvin
    W_M_DELTA_K
        watt per metre delta kelvin
    """

    BTU_IT_IN_H_FT2_DELTA_F = "Btu[IT].in/(h.ft2.deltaF)"
    J_M_S_M2_DELTA_K = "J.m/(s.m2.deltaK)"
    K_J_M_H_M2_DELTA_K = "kJ.m/(h.m2.deltaK)"
    W_M_DELTA_K = "W/(m.deltaK)"


class EnergyPerAreaUom(Enum):
    """
    Properties
    ----------
    ERG_CM2
        erg per square centimetre
    J_CM2
        joule per square centimetre
    J_M2
        joule per square metre
    KGF_M_CM2
        thousand gram-force metre per square centimetre
    LBF_FT_IN2
        foot pound-force per square inch
    M_J_CM2
        millijoule per square centimetre
    M_J_M2
        millijoule per square metre
    N_M
        newton per metre
    """

    ERG_CM2 = "erg/cm2"
    J_CM2 = "J/cm2"
    J_M2 = "J/m2"
    KGF_M_CM2 = "kgf.m/cm2"
    LBF_FT_IN2 = "lbf.ft/in2"
    M_J_CM2 = "mJ/cm2"
    M_J_M2 = "mJ/m2"
    N_M = "N/m"


class EnergyPerLengthUom(Enum):
    """
    Properties
    ----------
    J_M
        joule per metre
    MJ_M
        megajoule per metre
    """

    J_M = "J/m"
    MJ_M = "MJ/m"


class EnergyPerMassPerTimeUom(Enum):
    """
    Properties
    ----------
    MREM_H
        thousandth of irem per hour
    M_SV_H
        millisievert per hour
    REM_H
        rem per hour
    SV_H
        sievert per hour
    SV_S
        sievert per second
    """

    MREM_H = "mrem/h"
    M_SV_H = "mSv/h"
    REM_H = "rem/h"
    SV_H = "Sv/h"
    SV_S = "Sv/s"


class EnergyPerMassUom(Enum):
    """
    Properties
    ----------
    BTU_IT_LBM
        BTU per pound-mass
    CAL_TH_G
        calorie per gram
    CAL_TH_KG
        calorie per kilogram
    CAL_TH_LBM
        calorie per pound-mass
    ERG_G
        erg per gram
    ERG_KG
        erg per kilogram
    HP_H_LBM
        horsepower hour per pound-mass
    J_G
        joule per gram
    J_KG
        joule per kilogram
    KCAL_TH_G
        thousand calorie per gram
    KCAL_TH_KG
        thousand calorie per kilogram
    K_J_KG
        kilojoule per kilogram
    K_W_H_KG
        kilowatt hour per kilogram
    LBF_FT_LBM
        foot pound-force per pound-mass
    MJ_KG
        megajoule per kilogram
    MW_H_KG
        megawatt hour per kilogram
    """

    BTU_IT_LBM = "Btu[IT]/lbm"
    CAL_TH_G = "cal[th]/g"
    CAL_TH_KG = "cal[th]/kg"
    CAL_TH_LBM = "cal[th]/lbm"
    ERG_G = "erg/g"
    ERG_KG = "erg/kg"
    HP_H_LBM = "hp.h/lbm"
    J_G = "J/g"
    J_KG = "J/kg"
    KCAL_TH_G = "kcal[th]/g"
    KCAL_TH_KG = "kcal[th]/kg"
    K_J_KG = "kJ/kg"
    K_W_H_KG = "kW.h/kg"
    LBF_FT_LBM = "lbf.ft/lbm"
    MJ_KG = "MJ/kg"
    MW_H_KG = "MW.h/kg"


class EnergyPerVolumeUom(Enum):
    """
    Properties
    ----------
    BTU_IT_BBL
        BTU per barrel
    BTU_IT_FT3
        BTU per cubic foot
    BTU_IT_GAL_UK
        BTU per UK gallon
    BTU_IT_GAL_US
        BTU per US gallon
    CAL_TH_CM3
        calorie per cubic centimetre
    CAL_TH_M_L
        calorie per millilitre
    CAL_TH_MM3
        calorie per cubic millimetre
    ERG_CM3
        erg per cubic centimetre
    ERG_M3
        erg per cubic metre
    HP_H_BBL
        horsepower hour per barrel
    J_DM3
        joule per cubic decimetre
    J_M3
        joule per cubic metre
    KCAL_TH_CM3
        thousand calorie per cubic centimetre
    KCAL_TH_M3
        thousand calorie per cubic metre
    K_J_DM3
        kilojoule per cubic decimetre
    K_J_M3
        kilojoule per cubic metre
    K_W_H_DM3
        kilowatt hour per cubic decimetre
    K_W_H_M3
        kilowatt hour per cubic metre
    LBF_FT_BBL
        foot pound-force per barrel
    LBF_FT_GAL_US
        foot pound-force per US gallon
    MJ_M3
        megajoule per cubic metre
    MW_H_M3
        megawatt hour per cubic metre
    TONF_US_MI_BBL
        US ton-force mile per barrel
    """

    BTU_IT_BBL = "Btu[IT]/bbl"
    BTU_IT_FT3 = "Btu[IT]/ft3"
    BTU_IT_GAL_UK = "Btu[IT]/gal[UK]"
    BTU_IT_GAL_US = "Btu[IT]/gal[US]"
    CAL_TH_CM3 = "cal[th]/cm3"
    CAL_TH_M_L = "cal[th]/mL"
    CAL_TH_MM3 = "cal[th]/mm3"
    ERG_CM3 = "erg/cm3"
    ERG_M3 = "erg/m3"
    HP_H_BBL = "hp.h/bbl"
    J_DM3 = "J/dm3"
    J_M3 = "J/m3"
    KCAL_TH_CM3 = "kcal[th]/cm3"
    KCAL_TH_M3 = "kcal[th]/m3"
    K_J_DM3 = "kJ/dm3"
    K_J_M3 = "kJ/m3"
    K_W_H_DM3 = "kW.h/dm3"
    K_W_H_M3 = "kW.h/m3"
    LBF_FT_BBL = "lbf.ft/bbl"
    LBF_FT_GAL_US = "lbf.ft/gal[US]"
    MJ_M3 = "MJ/m3"
    MW_H_M3 = "MW.h/m3"
    TONF_US_MI_BBL = "tonf[US].mi/bbl"


class EnergyUom(Enum):
    """
    Properties
    ----------
    VALUE_1_E6_BTU_IT
        million BTU
    A_J
        attojoule
    BTU_IT
        British thermal unit
    BTU_TH
        thermochemical British thermal unit
    BTU_UK
        United Kingdom British thermal unit
    CAL_IT
        calorie [International Table]
    CAL_TH
        calorie
    CCAL_TH
        hundredth of calorie
    CE_V
        centielectronvolt
    C_J
        centijoule
    DCAL_TH
        tenth of calorie
    DE_V
        decielectronvolt
    D_J
        decijoule
    ECAL_TH
        million million million calorie
    EE_V
        exaelectronvolt
    EJ
        exajoule
    ERG
        erg
    E_V
        electronvolt
    FCAL_TH
        femtocalorie
    FE_V
        femtoelectronvolt
    F_J
        femtojoule
    GCAL_TH
        thousand million calorie
    GE_V
        gigaelectronvolt
    GJ
        gigajoule
    GW_H
        gigawatt hour
    HP_H
        horsepower hour
    HP_METRIC_H
        metric-horsepower hour
    J
        joule
    KCAL_TH
        thousand calorie
    KE_V
        kiloelectronvolt
    K_J
        kilojoule
    K_W_H
        kilowatt hour
    MCAL_TH
        million calorie
    MCAL_TH_1
        thousandth of calorie
    ME_V
        millielectronvolt
    ME_V_1
        megaelectronvolt
    MJ
        megajoule
    M_J_1
        millijoule
    MW_H
        megawatt hour
    NCAL_TH
        nanocalorie
    NE_V
        nanoelectronvolt
    N_J
        nanojoule
    PCAL_TH
        picocalorie
    PE_V
        picoelectronvolt
    P_J
        picojoule
    QUAD
        quad
    TCAL_TH
        million million calorie
    TE_V
        teraelectronvolt
    THERM_EC
        European Community therm
    THERM_UK
        United Kingdom therm
    THERM_US
        United States therm
    TJ
        terajoule
    TW_H
        terrawatt hour
    UCAL_TH
        millionth of calorie
    UE_V
        microelectronvolt
    U_J
        microjoule
    """

    VALUE_1_E6_BTU_IT = "1E6 Btu[IT]"
    A_J = "aJ"
    BTU_IT = "Btu[IT]"
    BTU_TH = "Btu[th]"
    BTU_UK = "Btu[UK]"
    CAL_IT = "cal[IT]"
    CAL_TH = "cal[th]"
    CCAL_TH = "ccal[th]"
    CE_V = "ceV"
    C_J = "cJ"
    DCAL_TH = "dcal[th]"
    DE_V = "deV"
    D_J = "dJ"
    ECAL_TH = "Ecal[th]"
    EE_V = "EeV"
    EJ = "EJ"
    ERG = "erg"
    E_V = "eV"
    FCAL_TH = "fcal[th]"
    FE_V = "feV"
    F_J = "fJ"
    GCAL_TH = "Gcal[th]"
    GE_V = "GeV"
    GJ = "GJ"
    GW_H = "GW.h"
    HP_H = "hp.h"
    HP_METRIC_H = "hp[metric].h"
    J = "J"
    KCAL_TH = "kcal[th]"
    KE_V = "keV"
    K_J = "kJ"
    K_W_H = "kW.h"
    MCAL_TH = "Mcal[th]"
    MCAL_TH_1 = "mcal[th]"
    ME_V = "meV"
    ME_V_1 = "MeV"
    MJ = "MJ"
    M_J_1 = "mJ"
    MW_H = "MW.h"
    NCAL_TH = "ncal[th]"
    NE_V = "neV"
    N_J = "nJ"
    PCAL_TH = "pcal[th]"
    PE_V = "peV"
    P_J = "pJ"
    QUAD = "quad"
    TCAL_TH = "Tcal[th]"
    TE_V = "TeV"
    THERM_EC = "therm[EC]"
    THERM_UK = "therm[UK]"
    THERM_US = "therm[US]"
    TJ = "TJ"
    TW_H = "TW.h"
    UCAL_TH = "ucal[th]"
    UE_V = "ueV"
    U_J = "uJ"


class ForceAreaUom(Enum):
    """
    Properties
    ----------
    DYNE_CM2
        dyne square centimetre
    KGF_M2
        thousand gram-force square metre
    K_N_M2
        kilonewton square metre
    LBF_IN2
        pound-force square inch
    M_N_M2
        millinewton square metre
    N_M2
        newton square metre
    PDL_CM2
        poundal square centimetre
    TONF_UK_FT2
        UK ton-force square foot
    TONF_US_FT2
        US ton-force square foot
    """

    DYNE_CM2 = "dyne.cm2"
    KGF_M2 = "kgf.m2"
    K_N_M2 = "kN.m2"
    LBF_IN2 = "lbf.in2"
    M_N_M2 = "mN.m2"
    N_M2 = "N.m2"
    PDL_CM2 = "pdl.cm2"
    TONF_UK_FT2 = "tonf[UK].ft2"
    TONF_US_FT2 = "tonf[US].ft2"


class ForceLengthPerLengthUom(Enum):
    """
    Properties
    ----------
    KGF_M_M
        thousand gram-force metre per metre
    LBF_FT_IN
        foot pound-force per inch
    LBF_IN_IN
        pound-force inch per inch
    N_M_M
        newton metre per metre
    TONF_US_MI_FT
        US ton-force mile per foot
    """

    KGF_M_M = "kgf.m/m"
    LBF_FT_IN = "lbf.ft/in"
    LBF_IN_IN = "lbf.in/in"
    N_M_M = "N.m/m"
    TONF_US_MI_FT = "tonf[US].mi/ft"


class ForcePerForceUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    EUC
        euclid
    KGF_KGF
        thousand gram-force per kilogram-force
    LBF_LBF
        pound-force per pound-force
    N_N
        newton per newton
    """

    VALUE = "%"
    EUC = "Euc"
    KGF_KGF = "kgf/kgf"
    LBF_LBF = "lbf/lbf"
    N_N = "N/N"


class ForcePerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_LBF_FT
        pound-force per hundred foot
    VALUE_1_30_LBF_M
        pound-force per thirty metre
    VALUE_1_30_N_M
        newton per thirty metre
    DYNE_CM
        dyne per centimetre
    KGF_CM
        thousand gram-force per centimetre
    K_N_M
        kilonewton per metre
    LBF_FT
        pound-force per foot
    LBF_IN
        pound-force per inch
    M_N_KM
        millinewton per kilometre
    M_N_M
        millinewton per metre
    N_M
        newton per metre
    PDL_CM
        poundal per centimetre
    TONF_UK_FT
        UK ton-force per foot
    TONF_US_FT
        US ton-force per foot
    """

    VALUE_0_01_LBF_FT = "0.01 lbf/ft"
    VALUE_1_30_LBF_M = "1/30 lbf/m"
    VALUE_1_30_N_M = "1/30 N/m"
    DYNE_CM = "dyne/cm"
    KGF_CM = "kgf/cm"
    K_N_M = "kN/m"
    LBF_FT = "lbf/ft"
    LBF_IN = "lbf/in"
    M_N_KM = "mN/km"
    M_N_M = "mN/m"
    N_M = "N/m"
    PDL_CM = "pdl/cm"
    TONF_UK_FT = "tonf[UK]/ft"
    TONF_US_FT = "tonf[US]/ft"


class ForcePerVolumeUom(Enum):
    """
    Properties
    ----------
    VALUE_0_001_PSI_FT
        psi per thousand foot
    VALUE_0_01_PSI_FT
        psi per hundred foot
    ATM_FT
        standard atmosphere per foot
    ATM_HM
        standard atmosphere per hundred metre
    ATM_M
        standard atmosphere per metre
    BAR_KM
        bar per kilometre
    BAR_M
        bar per metre
    GPA_CM
        gigapascal per centimetre
    K_PA_HM
        kilopascal per hectometre
    K_PA_M
        kilopascal per metre
    LBF_FT3
        pound-force per cubic foot
    LBF_GAL_US
        pound-force per US gallon
    MPA_M
        megapascal per metre
    N_M3
        newton per cubic metre
    PA_M
        pascal per metre
    PSI_FT
        psi per foot
    PSI_M
        psi per metre
    """

    VALUE_0_001_PSI_FT = "0.001 psi/ft"
    VALUE_0_01_PSI_FT = "0.01 psi/ft"
    ATM_FT = "atm/ft"
    ATM_HM = "atm/hm"
    ATM_M = "atm/m"
    BAR_KM = "bar/km"
    BAR_M = "bar/m"
    GPA_CM = "GPa/cm"
    K_PA_HM = "kPa/hm"
    K_PA_M = "kPa/m"
    LBF_FT3 = "lbf/ft3"
    LBF_GAL_US = "lbf/gal[US]"
    MPA_M = "MPa/m"
    N_M3 = "N/m3"
    PA_M = "Pa/m"
    PSI_FT = "psi/ft"
    PSI_M = "psi/m"


class ForceUom(Enum):
    """
    Properties
    ----------
    VALUE_10_K_N
        ten kilonewton
    C_N
        centinewton
    DA_N
        dekanewton
    D_N
        decinewton
    DYNE
        dyne
    EN
        exanewton
    F_N
        femtonewton
    GF
        gram-force
    GN
        giganewton
    H_N
        hectonewton
    KDYNE
        kilodyne
    KGF
        thousand gram-force
    KLBF
        thousand pound-force
    K_N
        kilonewton
    LBF
        pound-force
    MGF
        million gram-force
    M_N
        millinewton
    MN_1
        meganewton
    N
        newton
    N_N
        nanonewton
    OZF
        ounce-force
    PDL
        poundal
    P_N
        piconewton
    TN
        teranewton
    TONF_UK
        UK ton-force
    TONF_US
        US ton-force
    U_N
        micronewton
    """

    VALUE_10_K_N = "10 kN"
    C_N = "cN"
    DA_N = "daN"
    D_N = "dN"
    DYNE = "dyne"
    EN = "EN"
    F_N = "fN"
    GF = "gf"
    GN = "GN"
    H_N = "hN"
    KDYNE = "kdyne"
    KGF = "kgf"
    KLBF = "klbf"
    K_N = "kN"
    LBF = "lbf"
    MGF = "Mgf"
    M_N = "mN"
    MN_1 = "MN"
    N = "N"
    N_N = "nN"
    OZF = "ozf"
    PDL = "pdl"
    P_N = "pN"
    TN = "TN"
    TONF_UK = "tonf[UK]"
    TONF_US = "tonf[US]"
    U_N = "uN"


class FrequencyIntervalUom(Enum):
    """
    Properties
    ----------
    O
        octave
    """

    O = "O"


class FrequencyUom(Enum):
    """
    Properties
    ----------
    C_HZ
        centihertz
    D_HZ
        decihertz
    EHZ
        exahertz
    F_HZ
        femtohertz
    GHZ
        gigahertz
    HZ
        hertz
    K_HZ
        kilohertz
    M_HZ
        millihertz
    MHZ_1
        megahertz
    N_HZ
        nanohertz
    P_HZ
        picohertz
    THZ
        terahertz
    U_HZ
        microhertz
    """

    C_HZ = "cHz"
    D_HZ = "dHz"
    EHZ = "EHz"
    F_HZ = "fHz"
    GHZ = "GHz"
    HZ = "Hz"
    K_HZ = "kHz"
    M_HZ = "mHz"
    MHZ_1 = "MHz"
    N_HZ = "nHz"
    P_HZ = "pHz"
    THZ = "THz"
    U_HZ = "uHz"


class HeatCapacityUom(Enum):
    """
    Properties
    ----------
    J_DELTA_K
        joule per delta kelvin
    """

    J_DELTA_K = "J/deltaK"


class HeatFlowRateUom(Enum):
    """
    Properties
    ----------
    VALUE_1_E6_BTU_IT_H
        million BTU per hour
    BTU_IT_H
        BTU per hour
    BTU_IT_MIN
        BTU per minute
    BTU_IT_S
        BTU per second
    CAL_TH_H
        calorie per hour
    EJ_A
        exajoule per julian-year
    ERG_A
        erg per julian-year
    GW
        gigawatt
    J_S
        joule per second
    KCAL_TH_H
        thousand calorie per hour
    K_W
        kilowatt
    LBF_FT_MIN
        foot pound-force per minute
    LBF_FT_S
        foot pound-force per second
    MJ_A
        megajoule per julian-year
    MW
        megawatt
    M_W_1
        milliwatt
    N_W
        nanowatt
    QUAD_A
        quad per julian-year
    TJ_A
        terajoule per julian-year
    TW
        terawatt
    UCAL_TH_S
        millionth of calorie per second
    U_W
        microwatt
    W
        watt
    """

    VALUE_1_E6_BTU_IT_H = "1E6 Btu[IT]/h"
    BTU_IT_H = "Btu[IT]/h"
    BTU_IT_MIN = "Btu[IT]/min"
    BTU_IT_S = "Btu[IT]/s"
    CAL_TH_H = "cal[th]/h"
    EJ_A = "EJ/a"
    ERG_A = "erg/a"
    GW = "GW"
    J_S = "J/s"
    KCAL_TH_H = "kcal[th]/h"
    K_W = "kW"
    LBF_FT_MIN = "lbf.ft/min"
    LBF_FT_S = "lbf.ft/s"
    MJ_A = "MJ/a"
    MW = "MW"
    M_W_1 = "mW"
    N_W = "nW"
    QUAD_A = "quad/a"
    TJ_A = "TJ/a"
    TW = "TW"
    UCAL_TH_S = "ucal[th]/s"
    U_W = "uW"
    W = "W"


class HeatTransferCoefficientUom(Enum):
    """
    Properties
    ----------
    BTU_IT_H_FT2_DELTA_F
        BTU per hour square foot delta Fahrenheit
    BTU_IT_H_FT2_DELTA_R
        BTU per hour square foot delta Rankine
    BTU_IT_H_M2_DELTA_C
        BTU per hour square metre delta Celsius
    BTU_IT_S_FT2_DELTA_F
        (BTU per second) per square foot delta Fahrenheit
    CAL_TH_H_CM2_DELTA_C
        calorie per hour square centimetre delta Celsius
    CAL_TH_S_CM2_DELTA_C
        calorie per second square centimetre delta Celsius
    J_S_M2_DELTA_C
        joule per second square metre delta Celsius
    KCAL_TH_H_M2_DELTA_C
        thousand calorie per hour square metre delta Celsius
    K_J_H_M2_DELTA_K
        kilojoule per hour square metre delta kelvin
    K_W_M2_DELTA_K
        kilowatt per square metre delta kelvin
    W_M2_DELTA_K
        watt per square metre delta kelvin
    """

    BTU_IT_H_FT2_DELTA_F = "Btu[IT]/(h.ft2.deltaF)"
    BTU_IT_H_FT2_DELTA_R = "Btu[IT]/(h.ft2.deltaR)"
    BTU_IT_H_M2_DELTA_C = "Btu[IT]/(h.m2.deltaC)"
    BTU_IT_S_FT2_DELTA_F = "Btu[IT]/(s.ft2.deltaF)"
    CAL_TH_H_CM2_DELTA_C = "cal[th]/(h.cm2.deltaC)"
    CAL_TH_S_CM2_DELTA_C = "cal[th]/(s.cm2.deltaC)"
    J_S_M2_DELTA_C = "J/(s.m2.deltaC)"
    KCAL_TH_H_M2_DELTA_C = "kcal[th]/(h.m2.deltaC)"
    K_J_H_M2_DELTA_K = "kJ/(h.m2.deltaK)"
    K_W_M2_DELTA_K = "kW/(m2.deltaK)"
    W_M2_DELTA_K = "W/(m2.deltaK)"


class IlluminanceUom(Enum):
    """
    Properties
    ----------
    FOOTCANDLE
        footcandle
    KLX
        kilolux
    LM_M2
        lumen per square metre
    LX
        lux
    """

    FOOTCANDLE = "footcandle"
    KLX = "klx"
    LM_M2 = "lm/m2"
    LX = "lx"


class InductanceUom(Enum):
    """
    Properties
    ----------
    C_H
        centihenry
    D_H
        decihenry
    EH
        exahenry
    F_H
        femtohenry
    GH
        gigahenry
    H
        henry
    K_H
        kilohenry
    MH
        megahenry
    M_H_1
        millihenry
    N_H
        nanohenry
    TH
        terahenry
    U_H
        microhenry
    """

    C_H = "cH"
    D_H = "dH"
    EH = "EH"
    F_H = "fH"
    GH = "GH"
    H = "H"
    K_H = "kH"
    MH = "MH"
    M_H_1 = "mH"
    N_H = "nH"
    TH = "TH"
    U_H = "uH"


class IsothermalCompressibilityUom(Enum):
    """
    Properties
    ----------
    DM3_K_W_H
        cubic decimetre per kilowatt hour
    DM3_MJ
        cubic decimetre per megajoule
    M3_K_W_H
        cubic metre per kilowatt hour
    M3_J
        cubic metre per joule
    MM3_J
        cubic millimetre per joule
    PT_UK_HP_H
        UK pint per horsepower hour
    """

    DM3_K_W_H = "dm3/(kW.h)"
    DM3_MJ = "dm3/MJ"
    M3_K_W_H = "m3/(kW.h)"
    M3_J = "m3/J"
    MM3_J = "mm3/J"
    PT_UK_HP_H = "pt[UK]/(hp.h)"


class KinematicViscosityUom(Enum):
    """
    Properties
    ----------
    CM2_S
        square centimetre per second
    C_ST
        centistokes
    FT2_H
        square foot per hour
    FT2_S
        square foot per second
    IN2_S
        square inch per second
    M2_H
        square metre per hour
    M2_S
        square metre per second
    MM2_S
        square millimetre per second
    PA_S_M3_KG
        pascal second square metre per kilogram
    ST
        stokes
    """

    CM2_S = "cm2/s"
    C_ST = "cSt"
    FT2_H = "ft2/h"
    FT2_S = "ft2/s"
    IN2_S = "in2/s"
    M2_H = "m2/h"
    M2_S = "m2/s"
    MM2_S = "mm2/s"
    PA_S_M3_KG = "Pa.s.m3/kg"
    ST = "St"


class LengthPerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    VALUE_0_01_FT_FT
        foot per hundred foot
    VALUE_1_30_M_M
        metre per thirty metre
    EUC
        euclid
    FT_FT
        foot per foot
    FT_IN
        foot per inch
    FT_M
        foot per metre
    FT_MI
        foot per mile
    KM_CM
        kilometre per centimetre
    M_CM
        metre per centimetre
    M_KM
        metre per kilometre
    M_M
        metre per metre
    MI_IN
        mile per inch
    """

    VALUE = "%"
    VALUE_0_01_FT_FT = "0.01 ft/ft"
    VALUE_1_30_M_M = "1/30 m/m"
    EUC = "Euc"
    FT_FT = "ft/ft"
    FT_IN = "ft/in"
    FT_M = "ft/m"
    FT_MI = "ft/mi"
    KM_CM = "km/cm"
    M_CM = "m/cm"
    M_KM = "m/km"
    M_M = "m/m"
    MI_IN = "mi/in"


class LengthPerMassUom(Enum):
    """
    Properties
    ----------
    FT_LBM
        foot per pound-mass
    M_KG
        metre per kilogram
    """

    FT_LBM = "ft/lbm"
    M_KG = "m/kg"


class LengthPerPressureUom(Enum):
    """
    Properties
    ----------
    FT_PSI
        foot per psi
    M_K_PA
        metre per kilopascal
    M_PA
        metre per Pascal
    """

    FT_PSI = "ft/psi"
    M_K_PA = "m/kPa"
    M_PA = "m/Pa"


class LengthPerTemperatureUom(Enum):
    """
    Properties
    ----------
    FT_DELTA_F
        foot per delta Fahrenheit
    M_DELTA_K
        metre per delta kelvin
    """

    FT_DELTA_F = "ft/deltaF"
    M_DELTA_K = "m/deltaK"


class LengthPerTimeUom(Enum):
    """
    Properties
    ----------
    VALUE_1000_FT_H
        thousand foot per hour
    VALUE_1000_FT_S
        thousand foot per second
    CM_A
        centimetre per julian-year
    CM_S
        centimetre per second
    DM_S
        decimetre per second
    FT_D
        foot per day
    FT_H
        foot per hour
    FT_MIN
        foot per minute
    FT_MS
        foot per millisecond
    FT_S
        foot per second
    FT_US
        foot per microsecond
    IN_A
        inch per julian-year
    IN_MIN
        inch per minute
    IN_S
        inch per second
    KM_H
        kilometre per hour
    KM_S
        kilometre per second
    KNOT
        knot
    M_D
        metre per day
    M_H
        metre per hour
    M_MIN
        metre per minute
    M_MS
        metre per millisecond
    M_S
        metre per second
    MI_H
        mile per hour
    MIL_A
        mil per julian-year
    MM_A
        millimetre per julian-year
    MM_S_1
        millimetre per second
    NM_S
        nanometre per second
    UM_S
        micrometre per second
    """

    VALUE_1000_FT_H = "1000 ft/h"
    VALUE_1000_FT_S = "1000 ft/s"
    CM_A = "cm/a"
    CM_S = "cm/s"
    DM_S = "dm/s"
    FT_D = "ft/d"
    FT_H = "ft/h"
    FT_MIN = "ft/min"
    FT_MS = "ft/ms"
    FT_S = "ft/s"
    FT_US = "ft/us"
    IN_A = "in/a"
    IN_MIN = "in/min"
    IN_S = "in/s"
    KM_H = "km/h"
    KM_S = "km/s"
    KNOT = "knot"
    M_D = "m/d"
    M_H = "m/h"
    M_MIN = "m/min"
    M_MS = "m/ms"
    M_S = "m/s"
    MI_H = "mi/h"
    MIL_A = "mil/a"
    MM_A = "mm/a"
    MM_S_1 = "mm/s"
    NM_S = "nm/s"
    UM_S = "um/s"


class LengthPerVolumeUom(Enum):
    """
    Properties
    ----------
    FT_BBL
        foot per barrel
    FT_FT3
        foot per cubic foot
    FT_GAL_US
        foot per US gallon
    KM_DM3
        kilometre per cubic decimetre
    KM_L
        kilometre per litre
    M_M3
        metre per cubic metre
    MI_GAL_UK
        mile per UK gallon
    MI_GAL_US
        mile per US gallon
    """

    FT_BBL = "ft/bbl"
    FT_FT3 = "ft/ft3"
    FT_GAL_US = "ft/gal[US]"
    KM_DM3 = "km/dm3"
    KM_L = "km/L"
    M_M3 = "m/m3"
    MI_GAL_UK = "mi/gal[UK]"
    MI_GAL_US = "mi/gal[US]"


class LengthUom(Enum):
    """
    Properties
    ----------
    VALUE_0_1_FT
        tenth of foot
    VALUE_0_1_FT_US
        tenth of US survey foot
    VALUE_0_1_IN
        tenth of inch
    VALUE_0_1_YD
        tenth of yard
    VALUE_1_16_IN
        sixteenth of inch
    VALUE_1_2_FT
        half of Foot
    VALUE_1_32_IN
        thirty-second of inch
    VALUE_1_64_IN
        sixty-fourth of inch
    VALUE_10_FT
        ten foot
    VALUE_10_IN
        ten inch
    VALUE_10_KM
        10 kilometre
    VALUE_100_FT
        hundred foot
    VALUE_100_KM
        100 kilometre
    VALUE_1000_FT
        thousand foot
    VALUE_30_FT
        thirty foot
    VALUE_30_M
        thirty metres
    ANGSTROM
        angstrom
    CHAIN
        chain
    CHAIN_BN_A
        British chain [Benoit 1895 A]
    CHAIN_BN_B
        British chain [Benoit 1895 B]
    CHAIN_CLA
        Clarke chain
    CHAIN_IND37
        Indian Chain [1937]
    CHAIN_SE
        British chain [Sears 1922]
    CHAIN_SE_T
        British chain [Sears 1922 truncated]
    CHAIN_US
        US survey chain
    CM
        centimetre
    DAM
        dekametre
    DM
        decimetre
    EM
        exametre
    FATHOM
        international fathom
    FM
        femtometre
    FT
        foot
    FT_BN_A
        British foot [Benoit 1895 A]
    FT_BN_B
        British foot [Benoit 1895 B]
    FT_BR36
        British foot [1936]
    FT_BR65
        British foot [1865]
    FT_CLA
        Clarke foot
    FT_GC
        Gold Coast foot
    FT_IND
        indian foot
    FT_IND37
        indian foot [1937]
    FT_IND62
        indian foot ]1962]
    FT_IND75
        indian foot [1975]
    FT_SE
        British foot [Sears 1922]
    FT_SE_T
        British foot [Sears 1922 truncated]
    FT_US
        US survey foot
    FUR_US
        furlong US survey
    GM
        gigametre
    HM
        hectometre
    IN
        inch
    IN_US
        US survey inch
    KM
        kilometre
    LINK
        link
    LINK_BN_A
        British link [Benoit 1895 A]
    LINK_BN_B
        British link [Benoit 1895 B]
    LINK_CLA
        Clarke link
    LINK_SE
        British link [Sears 1922]
    LINK_SE_T
        British link [Sears 1922 truncated]
    LINK_US
        US survey link
    M
        metre
    M_GER
        German legal metre
    MI
        mile
    MI_NAUT
        international nautical mile
    MI_NAUT_UK
        United Kingdom nautical mile
    MI_US
        US survey mile
    MIL
        mil
    MM
        megametre
    MM_1
        millimetre
    NM
        nanometre
    PM
        picometre
    ROD_US
        rod US Survey
    TM
        terametre
    UM
        micrometre
    YD
        yard
    YD_BN_A
        British yard [Benoit 1895 A]
    YD_BN_B
        British yard [Benoit 1895 B]
    YD_CLA
        Clarke yard
    YD_IND
        Indian yard
    YD_IND37
        Indian yard [1937]
    YD_IND62
        Indian yard [1962]
    YD_IND75
        Indian yard [1975]
    YD_SE
        British yard [Sears 1922]
    YD_SE_T
        British yard [Sears 1922 truncated]
    YD_US
        US survey yard
    """

    VALUE_0_1_FT = "0.1 ft"
    VALUE_0_1_FT_US = "0.1 ft[US]"
    VALUE_0_1_IN = "0.1 in"
    VALUE_0_1_YD = "0.1 yd"
    VALUE_1_16_IN = "1/16 in"
    VALUE_1_2_FT = "1/2 ft"
    VALUE_1_32_IN = "1/32 in"
    VALUE_1_64_IN = "1/64 in"
    VALUE_10_FT = "10 ft"
    VALUE_10_IN = "10 in"
    VALUE_10_KM = "10 km"
    VALUE_100_FT = "100 ft"
    VALUE_100_KM = "100 km"
    VALUE_1000_FT = "1000 ft"
    VALUE_30_FT = "30 ft"
    VALUE_30_M = "30 m"
    ANGSTROM = "angstrom"
    CHAIN = "chain"
    CHAIN_BN_A = "chain[BnA]"
    CHAIN_BN_B = "chain[BnB]"
    CHAIN_CLA = "chain[Cla]"
    CHAIN_IND37 = "chain[Ind37]"
    CHAIN_SE = "chain[Se]"
    CHAIN_SE_T = "chain[SeT]"
    CHAIN_US = "chain[US]"
    CM = "cm"
    DAM = "dam"
    DM = "dm"
    EM = "Em"
    FATHOM = "fathom"
    FM = "fm"
    FT = "ft"
    FT_BN_A = "ft[BnA]"
    FT_BN_B = "ft[BnB]"
    FT_BR36 = "ft[Br36]"
    FT_BR65 = "ft[Br65]"
    FT_CLA = "ft[Cla]"
    FT_GC = "ft[GC]"
    FT_IND = "ft[Ind]"
    FT_IND37 = "ft[Ind37]"
    FT_IND62 = "ft[Ind62]"
    FT_IND75 = "ft[Ind75]"
    FT_SE = "ft[Se]"
    FT_SE_T = "ft[SeT]"
    FT_US = "ft[US]"
    FUR_US = "fur[US]"
    GM = "Gm"
    HM = "hm"
    IN = "in"
    IN_US = "in[US]"
    KM = "km"
    LINK = "link"
    LINK_BN_A = "link[BnA]"
    LINK_BN_B = "link[BnB]"
    LINK_CLA = "link[Cla]"
    LINK_SE = "link[Se]"
    LINK_SE_T = "link[SeT]"
    LINK_US = "link[US]"
    M = "m"
    M_GER = "m[Ger]"
    MI = "mi"
    MI_NAUT = "mi[naut]"
    MI_NAUT_UK = "mi[nautUK]"
    MI_US = "mi[US]"
    MIL = "mil"
    MM = "Mm"
    MM_1 = "mm"
    NM = "nm"
    PM = "pm"
    ROD_US = "rod[US]"
    TM = "Tm"
    UM = "um"
    YD = "yd"
    YD_BN_A = "yd[BnA]"
    YD_BN_B = "yd[BnB]"
    YD_CLA = "yd[Cla]"
    YD_IND = "yd[Ind]"
    YD_IND37 = "yd[Ind37]"
    YD_IND62 = "yd[Ind62]"
    YD_IND75 = "yd[Ind75]"
    YD_SE = "yd[Se]"
    YD_SE_T = "yd[SeT]"
    YD_US = "yd[US]"


class LightExposureUom(Enum):
    """
    Properties
    ----------
    FOOTCANDLE_S
        footcandle second
    LX_S
        lux second
    """

    FOOTCANDLE_S = "footcandle.s"
    LX_S = "lx.s"


class LinearAccelerationUom(Enum):
    """
    Properties
    ----------
    CM_S2
        centimetre per square second
    FT_S2
        foot per second squared
    GAL
        galileo
    GN
        gravity
    IN_S2
        inch per second squared
    M_S2
        metre per second squared
    M_GAL
        milligalileo
    MGN
        thousandth of gravity
    """

    CM_S2 = "cm/s2"
    FT_S2 = "ft/s2"
    GAL = "Gal"
    GN = "gn"
    IN_S2 = "in/s2"
    M_S2 = "m/s2"
    M_GAL = "mGal"
    MGN = "mgn"


class LinearThermalExpansionUom(Enum):
    """
    Properties
    ----------
    VALUE_1_DELTA_K
        per delta kelvin
    IN_IN_DELTA_F
        inch per inch delta Fahrenheit
    M_M_DELTA_K
        metre per metre delta kelvin
    MM_MM_DELTA_K
        millimetre per millimetre delta kelvin
    """

    VALUE_1_DELTA_K = "1/deltaK"
    IN_IN_DELTA_F = "in/(in.deltaF)"
    M_M_DELTA_K = "m/(m.deltaK)"
    MM_MM_DELTA_K = "mm/(mm.deltaK)"


class LogarithmicPowerRatioPerLengthUom(Enum):
    """
    Properties
    ----------
    B_M
        bel per metre
    D_B_FT
        decibel per foot
    D_B_KM
        decibel per kilometre
    D_B_M
        decibel per metre
    """

    B_M = "B/m"
    D_B_FT = "dB/ft"
    D_B_KM = "dB/km"
    D_B_M = "dB/m"


class LogarithmicPowerRatioUom(Enum):
    """
    Properties
    ----------
    B
        bel
    D_B
        decibel
    """

    B = "B"
    D_B = "dB"


class LuminanceUom(Enum):
    """
    Properties
    ----------
    CD_M2
        candela per square metre
    """

    CD_M2 = "cd/m2"


class LuminousEfficacyUom(Enum):
    """
    Properties
    ----------
    LM_W
        lumen per watt
    """

    LM_W = "lm/W"


class LuminousFluxUom(Enum):
    """
    Properties
    ----------
    LM
        lumen
    """

    LM = "lm"


class LuminousIntensityUom(Enum):
    """
    Properties
    ----------
    CD
        candela
    KCD
        kilocandela
    """

    CD = "cd"
    KCD = "kcd"


class MagneticDipoleMomentUom(Enum):
    """
    Properties
    ----------
    WB_M
        weber metre
    """

    WB_M = "Wb.m"


class MagneticFieldStrengthUom(Enum):
    """
    Properties
    ----------
    A_M
        ampere per metre
    A_MM
        ampere per millimetre
    OE
        oersted
    """

    A_M = "A/m"
    A_MM = "A/mm"
    OE = "Oe"


class MagneticFluxDensityPerLengthUom(Enum):
    """
    Properties
    ----------
    GAUSS_CM
        gauss per centimetre
    M_T_DM
        millitesla per decimetre
    T_M
        tesla per metre
    """

    GAUSS_CM = "gauss/cm"
    M_T_DM = "mT/dm"
    T_M = "T/m"


class MagneticFluxDensityUom(Enum):
    """
    Properties
    ----------
    CGAUSS
        centigauss
    C_T
        centitesla
    DGAUSS
        decigauss
    D_T
        decitesla
    EGAUSS
        exagauss
    ET
        exatesla
    FGAUSS
        femtogauss
    F_T
        femtotesla
    GAUSS
        gauss
    GGAUSS
        gigagauss
    GT
        gigatesla
    KGAUSS
        kilogauss
    K_T
        kilotesla
    MGAUSS
        milligauss
    MGAUSS_1
        megagauss
    M_T
        millitesla
    NGAUSS
        nanogauss
    N_T
        nanotesla
    PGAUSS
        picogauss
    P_T
        picotesla
    T
        tesla
    TGAUSS
        teragauss
    TT
        teratesla
    UGAUSS
        microgauss
    U_T
        microtesla
    """

    CGAUSS = "cgauss"
    C_T = "cT"
    DGAUSS = "dgauss"
    D_T = "dT"
    EGAUSS = "Egauss"
    ET = "ET"
    FGAUSS = "fgauss"
    F_T = "fT"
    GAUSS = "gauss"
    GGAUSS = "Ggauss"
    GT = "GT"
    KGAUSS = "kgauss"
    K_T = "kT"
    MGAUSS = "mgauss"
    MGAUSS_1 = "Mgauss"
    M_T = "mT"
    NGAUSS = "ngauss"
    N_T = "nT"
    PGAUSS = "pgauss"
    P_T = "pT"
    T = "T"
    TGAUSS = "Tgauss"
    TT = "TT"
    UGAUSS = "ugauss"
    U_T = "uT"


class MagneticFluxUom(Enum):
    """
    Properties
    ----------
    C_WB
        centiweber
    D_WB
        deciweber
    EWB
        exaweber
    F_WB
        femtoweber
    GWB
        gigaweber
    K_WB
        kiloweber
    MWB
        megaweber
    M_WB_1
        milliweber
    N_WB
        nanoweber
    P_WB
        picoweber
    TWB
        teraweber
    U_WB
        microweber
    WB
        weber
    """

    C_WB = "cWb"
    D_WB = "dWb"
    EWB = "EWb"
    F_WB = "fWb"
    GWB = "GWb"
    K_WB = "kWb"
    MWB = "MWb"
    M_WB_1 = "mWb"
    N_WB = "nWb"
    P_WB = "pWb"
    TWB = "TWb"
    U_WB = "uWb"
    WB = "Wb"


class MagneticPermeabilityUom(Enum):
    """
    Properties
    ----------
    H_M
        henry per metre
    U_H_M
        microhenry per metre
    """

    H_M = "H/m"
    U_H_M = "uH/m"


class MagneticVectorPotentialUom(Enum):
    """
    Properties
    ----------
    WB_M
        weber per metre
    WB_MM
        weber per millimetre
    """

    WB_M = "Wb/m"
    WB_MM = "Wb/mm"


class MassLengthUom(Enum):
    """
    Properties
    ----------
    KG_M
        kilogram metre
    LBM_FT
        pound-mass foot
    """

    KG_M = "kg.m"
    LBM_FT = "lbm.ft"


class MassPerAreaUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_LBM_FT2
        pound-mass per hundred square foot
    KG_M2
        kilogram per square metre
    LBM_FT2
        pound-mass per square foot
    MG_M2
        megagram per square metre
    TON_US_FT2
        US ton-mass per square foot
    """

    VALUE_0_01_LBM_FT2 = "0.01 lbm/ft2"
    KG_M2 = "kg/m2"
    LBM_FT2 = "lbm/ft2"
    MG_M2 = "Mg/m2"
    TON_US_FT2 = "ton[US]/ft2"


class MassPerEnergyUom(Enum):
    """
    Properties
    ----------
    KG_K_W_H
        kilogram per kilowatt hour
    KG_J
        kilogram per joule
    KG_MJ
        kilogram per megajoule
    LBM_HP_H
        pound-mass per horsepower hour
    MG_J
        milligram per joule
    """

    KG_K_W_H = "kg/(kW.h)"
    KG_J = "kg/J"
    KG_MJ = "kg/MJ"
    LBM_HP_H = "lbm/(hp.h)"
    MG_J = "mg/J"


class MassPerLengthUom(Enum):
    """
    Properties
    ----------
    KG_M_CM2
        kilogram metre per square centimetre
    KG_M
        kilogram per metre
    KLBM_IN
        thousand pound-mass per inch
    LBM_FT
        pound-mass per foot
    MG_IN
        megagram per inch
    """

    KG_M_CM2 = "kg.m/cm2"
    KG_M = "kg/m"
    KLBM_IN = "klbm/in"
    LBM_FT = "lbm/ft"
    MG_IN = "Mg/in"


class MassPerMassUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    MASS
        percent [mass basis]
    EUC
        euclid
    G_KG
        gram per kilogram
    G_T
        gram per tonne
    KG_KG
        kilogram per kilogram
    KG_SACK_94LBM
        kilogram per 94-pound-sack
    KG_T
        kilogram per tonne
    MG_G
        milligram per gram
    MG_KG
        milligram per kilogram
    NG_G
        nanogram per gram
    NG_MG
        nanogram per milligram
    PPK
        part per thousand
    PPM
        part per million
    PPM_MASS
        part per million [mass basis]
    UG_G
        microgram per gram
    UG_MG
        microgram per milligram
    """

    VALUE = "%"
    MASS = "%[mass]"
    EUC = "Euc"
    G_KG = "g/kg"
    G_T = "g/t"
    KG_KG = "kg/kg"
    KG_SACK_94LBM = "kg/sack[94lbm]"
    KG_T = "kg/t"
    MG_G = "mg/g"
    MG_KG = "mg/kg"
    NG_G = "ng/g"
    NG_MG = "ng/mg"
    PPK = "ppk"
    PPM = "ppm"
    PPM_MASS = "ppm[mass]"
    UG_G = "ug/g"
    UG_MG = "ug/mg"


class MassPerTimePerAreaUom(Enum):
    """
    Properties
    ----------
    G_FT_CM3_S
        gram foot per cubic centimetre second
    G_M_CM3_S
        gram metre per cubic centimetre second
    KG_M2_S
        kilogram per square metre second
    K_PA_S_M
        kilopascal second per metre
    LBM_FT2_H
        pound-mass per square foot hour
    LBM_FT2_S
        pound-mass per square foot second
    MPA_S_M
        megapascal second per metre
    """

    G_FT_CM3_S = "g.ft/(cm3.s)"
    G_M_CM3_S = "g.m/(cm3.s)"
    KG_M2_S = "kg/(m2.s)"
    K_PA_S_M = "kPa.s/m"
    LBM_FT2_H = "lbm/(ft2.h)"
    LBM_FT2_S = "lbm/(ft2.s)"
    MPA_S_M = "MPa.s/m"


class MassPerTimePerLengthUom(Enum):
    """
    Properties
    ----------
    KG_M_S
        kilogram per metre second
    LBM_FT_H
        pound-mass per hour foot
    LBM_FT_S
        pound-mass per second foot
    PA_S
        pascal second
    """

    KG_M_S = "kg/(m.s)"
    LBM_FT_H = "lbm/(ft.h)"
    LBM_FT_S = "lbm/(ft.s)"
    PA_S = "Pa.s"


class MassPerTimeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_E6_LBM_A
        million pound-mass per julian-year
    G_S
        gram per second
    KG_D
        kilogram per day
    KG_H
        kilogram per hour
    KG_MIN
        kilogram per min
    KG_S
        kilogram per second
    LBM_D
        pound-mass per day
    LBM_H
        pound-mass per hour
    LBM_MIN
        pound-mass per minute
    LBM_S
        pound-mass per second
    MG_A
        megagram per julian-year
    MG_D
        megagram per day
    MG_H
        megagram per hour
    MG_MIN
        megagram per minute
    T_A
        tonne per julian-year
    T_D
        tonne per day
    T_H
        tonne per hour
    T_MIN
        tonne per minute
    TON_UK_A
        UK ton-mass per julian-year
    TON_UK_D
        UK ton-mass per day
    TON_UK_H
        UK ton-mass per hour
    TON_UK_MIN
        UK ton-mass per minute
    TON_US_A
        US ton-mass per julian-year
    TON_US_D
        US ton-mass per day
    TON_US_H
        US ton-mass per hour
    TON_US_MIN
        US ton-mass per minute
    """

    VALUE_1_E6_LBM_A = "1E6 lbm/a"
    G_S = "g/s"
    KG_D = "kg/d"
    KG_H = "kg/h"
    KG_MIN = "kg/min"
    KG_S = "kg/s"
    LBM_D = "lbm/d"
    LBM_H = "lbm/h"
    LBM_MIN = "lbm/min"
    LBM_S = "lbm/s"
    MG_A = "Mg/a"
    MG_D = "Mg/d"
    MG_H = "Mg/h"
    MG_MIN = "Mg/min"
    T_A = "t/a"
    T_D = "t/d"
    T_H = "t/h"
    T_MIN = "t/min"
    TON_UK_A = "ton[UK]/a"
    TON_UK_D = "ton[UK]/d"
    TON_UK_H = "ton[UK]/h"
    TON_UK_MIN = "ton[UK]/min"
    TON_US_A = "ton[US]/a"
    TON_US_D = "ton[US]/d"
    TON_US_H = "ton[US]/h"
    TON_US_MIN = "ton[US]/min"


class MassPerVolumePerLengthUom(Enum):
    """
    Properties
    ----------
    G_CM4
        gram per centimetre to the fourth power
    KG_DM4
        kilogram per decimetre to the fourth power
    KG_M4
        kilogram per metre to the fourth power
    LBM_GAL_UK_FT
        pound-mass per UK gallon foot
    LBM_GAL_US_FT
        pound-mass per US gallon foot
    LBM_FT4
        pound-mass per foot to the fourth power
    PA_S2_M3
        pascal second squared per cubic metre
    """

    G_CM4 = "g/cm4"
    KG_DM4 = "kg/dm4"
    KG_M4 = "kg/m4"
    LBM_GAL_UK_FT = "lbm/(gal[UK].ft)"
    LBM_GAL_US_FT = "lbm/(gal[US].ft)"
    LBM_FT4 = "lbm/ft4"
    PA_S2_M3 = "Pa.s2/m3"


class MassPerVolumeUom(Enum):
    """
    Properties
    ----------
    VALUE_0_001_LBM_BBL
        pound-mass per thousand barrel
    VALUE_0_001_LBM_GAL_UK
        pound-mass per thousand UK gallon
    VALUE_0_001_LBM_GAL_US
        pound-mass per thousand US gallon
    VALUE_0_01_GRAIN_FT3
        grain per hundred cubic foot
    VALUE_0_1_LBM_BBL
        pound-mass per ten barrel
    VALUE_10_MG_M3
        ten thousand kilogram per cubic metre
    G_CM3
        gram per cubic centimetre
    G_DM3
        gram per cubic decimetre
    G_GAL_UK
        gram per UK gallon
    G_GAL_US
        gram per US gallon
    G_L
        gram per litre
    G_M3
        gram per cubic metre
    GRAIN_FT3
        grain per cubic foot
    GRAIN_GAL_US
        grain per US gallon
    KG_DM3
        kilogram per cubic decimetre
    KG_L
        kilogram per litre
    KG_M3
        kilogram per cubic metre
    LBM_BBL
        pound-mass per barrel
    LBM_FT3
        pound-mass per cubic foot
    LBM_GAL_UK
        pound-mass per UK gallon
    LBM_GAL_US
        pound-mass per US gallon
    LBM_IN3
        pound-mass per cubic inch
    MG_DM3
        milligram per cubic decimetre
    MG_GAL_US
        milligram per US gallon
    MG_L
        milligram per litre
    MG_M3
        milligram per cubic metre
    MG_M3_1
        megagram per cubic metre
    T_M3
        tonne per cubic metre
    UG_CM3
        microgram per cubic centimetre
    """

    VALUE_0_001_LBM_BBL = "0.001 lbm/bbl"
    VALUE_0_001_LBM_GAL_UK = "0.001 lbm/gal[UK]"
    VALUE_0_001_LBM_GAL_US = "0.001 lbm/gal[US]"
    VALUE_0_01_GRAIN_FT3 = "0.01 grain/ft3"
    VALUE_0_1_LBM_BBL = "0.1 lbm/bbl"
    VALUE_10_MG_M3 = "10 Mg/m3"
    G_CM3 = "g/cm3"
    G_DM3 = "g/dm3"
    G_GAL_UK = "g/gal[UK]"
    G_GAL_US = "g/gal[US]"
    G_L = "g/L"
    G_M3 = "g/m3"
    GRAIN_FT3 = "grain/ft3"
    GRAIN_GAL_US = "grain/gal[US]"
    KG_DM3 = "kg/dm3"
    KG_L = "kg/L"
    KG_M3 = "kg/m3"
    LBM_BBL = "lbm/bbl"
    LBM_FT3 = "lbm/ft3"
    LBM_GAL_UK = "lbm/gal[UK]"
    LBM_GAL_US = "lbm/gal[US]"
    LBM_IN3 = "lbm/in3"
    MG_DM3 = "mg/dm3"
    MG_GAL_US = "mg/gal[US]"
    MG_L = "mg/L"
    MG_M3 = "mg/m3"
    MG_M3_1 = "Mg/m3"
    T_M3 = "t/m3"
    UG_CM3 = "ug/cm3"


class MassUom(Enum):
    """
    Properties
    ----------
    AG
        attogram
    CG
        centigram
    CT
        carat
    CWT_UK
        UK hundredweight
    CWT_US
        US hundredweight
    EG
        exagram
    FG
        femtogram
    G
        gram
    GG
        gigagram
    GRAIN
        grain
    HG
        hectogram
    KG
        kilogram
    KLBM
        thousand pound-mass
    LBM
        pound-mass
    MG
        milligram
    MG_1
        megagram
    NG
        nanogram
    OZM
        ounce-mass
    OZM_TROY
        troy ounce-mass
    PG
        picogram
    SACK_94LBM
        94 pound-mass sack
    T
        tonne
    TG
        teragram
    TON_UK
        UK ton-mass
    TON_US
        US ton-mass
    UG
        microgram
    """

    AG = "ag"
    CG = "cg"
    CT = "ct"
    CWT_UK = "cwt[UK]"
    CWT_US = "cwt[US]"
    EG = "Eg"
    FG = "fg"
    G = "g"
    GG = "Gg"
    GRAIN = "grain"
    HG = "hg"
    KG = "kg"
    KLBM = "klbm"
    LBM = "lbm"
    MG = "mg"
    MG_1 = "Mg"
    NG = "ng"
    OZM = "ozm"
    OZM_TROY = "ozm[troy]"
    PG = "pg"
    SACK_94LBM = "sack[94lbm]"
    T = "t"
    TG = "Tg"
    TON_UK = "ton[UK]"
    TON_US = "ton[US]"
    UG = "ug"


class MobilityUom(Enum):
    """
    Properties
    ----------
    D_PA_S
        darcy per pascal second
    D_C_P
        darcy per centipoise
    M_D_FT2_LBF_S
        millidarcy square foot per pound-force second
    M_D_IN2_LBF_S
        millidarcy square inch per pound-force second
    M_D_PA_S
        millidarcy per pascal second
    M_D_C_P
        millidarcy per centipoise
    TD_API_PA_S
        teradarcy-API per pascal second
    """

    D_PA_S = "D/(Pa.s)"
    D_C_P = "D/cP"
    M_D_FT2_LBF_S = "mD.ft2/(lbf.s)"
    M_D_IN2_LBF_S = "mD.in2/(lbf.s)"
    M_D_PA_S = "mD/(Pa.s)"
    M_D_C_P = "mD/cP"
    TD_API_PA_S = "TD[API]/(Pa.s)"


class MolarEnergyUom(Enum):
    """
    Properties
    ----------
    BTU_IT_LBMOL
        BTU per pound-mass-mole
    J_MOL
        joule per gram-mole
    KCAL_TH_MOL
        thousand calorie per gram-mole
    K_J_KMOL
        kilojoule per kilogram-mole
    MJ_KMOL
        megajoule per kilogram-mole
    """

    BTU_IT_LBMOL = "Btu[IT]/lbmol"
    J_MOL = "J/mol"
    KCAL_TH_MOL = "kcal[th]/mol"
    K_J_KMOL = "kJ/kmol"
    MJ_KMOL = "MJ/kmol"


class MolarHeatCapacityUom(Enum):
    """
    Properties
    ----------
    BTU_IT_LBMOL_DELTA_F
        BTU per pound-mass-mole delta Fahrenheit
    CAL_TH_MOL_DELTA_C
        calorie per gram-mole delta Celsius
    J_MOL_DELTA_K
        joule per gram-mole delta kelvin
    K_J_KMOL_DELTA_K
        kilojoule per kilogram-mole delta kelvin
    """

    BTU_IT_LBMOL_DELTA_F = "Btu[IT]/(lbmol.deltaF)"
    CAL_TH_MOL_DELTA_C = "cal[th]/(mol.deltaC)"
    J_MOL_DELTA_K = "J/(mol.deltaK)"
    K_J_KMOL_DELTA_K = "kJ/(kmol.deltaK)"


class MolarVolumeUom(Enum):
    """
    Properties
    ----------
    DM3_KMOL
        cubic decimetre per kilogram-mole
    FT3_LBMOL
        cubic foot per pound-mass-mole
    L_KMOL
        litre per kilogram-mole
    L_MOL
        litre per gram-mole
    M3_KMOL
        cubic metre per kilogram-mole
    M3_MOL
        cubic metre per gram-mole
    """

    DM3_KMOL = "dm3/kmol"
    FT3_LBMOL = "ft3/lbmol"
    L_KMOL = "L/kmol"
    L_MOL = "L/mol"
    M3_KMOL = "m3/kmol"
    M3_MOL = "m3/mol"


class MolecularWeightUom(Enum):
    """
    Properties
    ----------
    G_MOL
        gram per mole
    KG_MOL
        kilogram per mole
    LBM_LBMOL
        pound-mass per pound-mole
    """

    G_MOL = "g/mol"
    KG_MOL = "kg/mol"
    LBM_LBMOL = "lbm/lbmol"


class MomentOfForceUom(Enum):
    """
    Properties
    ----------
    VALUE_1000_LBF_FT
        thousand foot pound-force
    DA_N_M
        dekanewton metre
    D_N_M
        decinewton metre
    J
        joule
    KGF_M
        thousand gram-force metre
    K_N_M
        kilonewton metre
    LBF_FT
        foot pound-force
    LBF_IN
        inch pound-force
    LBM_FT2_S2
        pound-mass square foot per second squared
    N_M
        newton metre
    PDL_FT
        foot poundal
    TONF_US_FT
        US ton-force foot
    TONF_US_MI
        US ton-force mile
    """

    VALUE_1000_LBF_FT = "1000 lbf.ft"
    DA_N_M = "daN.m"
    D_N_M = "dN.m"
    J = "J"
    KGF_M = "kgf.m"
    K_N_M = "kN.m"
    LBF_FT = "lbf.ft"
    LBF_IN = "lbf.in"
    LBM_FT2_S2 = "lbm.ft2/s2"
    N_M = "N.m"
    PDL_FT = "pdl.ft"
    TONF_US_FT = "tonf[US].ft"
    TONF_US_MI = "tonf[US].mi"


class MomentOfInertiaUom(Enum):
    """
    Properties
    ----------
    KG_M2
        kilogram square metre
    LBM_FT2
        pound-mass square foot
    """

    KG_M2 = "kg.m2"
    LBM_FT2 = "lbm.ft2"


class MomentumUom(Enum):
    """
    Properties
    ----------
    KG_M_S
        kilogram metre per second
    LBM_FT_S
        foot pound-mass per second
    """

    KG_M_S = "kg.m/s"
    LBM_FT_S = "lbm.ft/s"


class NormalizedPowerUom(Enum):
    """
    Properties
    ----------
    B_W
        bel watt
    D_B_MW
        decibel megawatt
    D_B_M_W_1
        decibel milliwatt
    D_B_W
        decibel watt
    """

    B_W = "B.W"
    D_B_MW = "dB.MW"
    D_B_M_W_1 = "dB.mW"
    D_B_W = "dB.W"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ObjectAlias:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    identifier: str = field(
        metadata={
            "name": "Identifier",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    description: Optional[str] = field(
        default=None,
        metadata={
            "name": "Description",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "min_length": 1,
            "max_length": 256,
            "white_space": "collapse",
        },
    )
    authority: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_length": 1,
            "max_length": 64,
            "white_space": "collapse",
        },
    )


class PermeabilityLengthUom(Enum):
    """
    Properties
    ----------
    D_FT
        darcy foot
    D_M
        darcy metre
    M_D_FT
        millidarcy foot
    M_D_M
        millidarcy metre
    TD_API_M
        teradarcy-API metre
    """

    D_FT = "D.ft"
    D_M = "D.m"
    M_D_FT = "mD.ft"
    M_D_M = "mD.m"
    TD_API_M = "TD[API].m"


class PermeabilityRockUom(Enum):
    """
    Properties
    ----------
    D
        darcy
    D_API
        darcy-API
    M_D
        millidarcy
    TD_API
        teradarcy-API
    """

    D = "D"
    D_API = "D[API]"
    M_D = "mD"
    TD_API = "TD[API]"


class PermittivityUom(Enum):
    """
    Properties
    ----------
    F_M
        farad per metre
    U_F_M
        microfarad per metre
    """

    F_M = "F/m"
    U_F_M = "uF/m"


class PlaneAngleUom(Enum):
    """
    Properties
    ----------
    VALUE_0_001_SECA
        angular millisecond
    CCGR
        centesimal-second
    CGR
        centesimal-minute
    DEGA
        angular degree
    GON
        gon
    KRAD
        kiloradian
    MILA
        angular mil
    MINA
        angular minute
    MRAD
        megaradian
    MRAD_1
        milliradian
    RAD
        radian
    REV
        revolution
    SECA
        angular second
    URAD
        microradian
    """

    VALUE_0_001_SECA = "0.001 seca"
    CCGR = "ccgr"
    CGR = "cgr"
    DEGA = "dega"
    GON = "gon"
    KRAD = "krad"
    MILA = "mila"
    MINA = "mina"
    MRAD = "Mrad"
    MRAD_1 = "mrad"
    RAD = "rad"
    REV = "rev"
    SECA = "seca"
    URAD = "urad"


class PotentialDifferencePerPowerDropUom(Enum):
    """
    Properties
    ----------
    V_B
        volt per bel
    V_D_B
        volt per decibel
    """

    V_B = "V/B"
    V_D_B = "V/dB"


class PowerPerAreaUom(Enum):
    """
    Properties
    ----------
    BTU_IT_H_FT2
        (BTU per hour) per square foot
    BTU_IT_S_FT2
        BTU per second square foot
    CAL_TH_H_CM2
        calorie per hour square centimetre
    HP_IN2
        horsepower per square inch
    HP_HYD_IN2
        hydraulic-horsepower per square inch
    K_W_CM2
        kilowatt per square centimetre
    K_W_M2
        kilowatt per square metre
    M_W_M2
        milliwatt per square metre
    UCAL_TH_S_CM2
        millionth of calorie per second square centimetre
    W_CM2
        watt per square centimetre
    W_M2
        watt per square metre
    W_MM2
        watt per square millimetre
    """

    BTU_IT_H_FT2 = "Btu[IT]/(h.ft2)"
    BTU_IT_S_FT2 = "Btu[IT]/(s.ft2)"
    CAL_TH_H_CM2 = "cal[th]/(h.cm2)"
    HP_IN2 = "hp/in2"
    HP_HYD_IN2 = "hp[hyd]/in2"
    K_W_CM2 = "kW/cm2"
    K_W_M2 = "kW/m2"
    M_W_M2 = "mW/m2"
    UCAL_TH_S_CM2 = "ucal[th]/(s.cm2)"
    W_CM2 = "W/cm2"
    W_M2 = "W/m2"
    W_MM2 = "W/mm2"


class PowerPerPowerUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    BTU_IT_HP_H
        BTU per horsepower hour
    EUC
        euclid
    W_K_W
        watt per kilowatt
    W_W
        watt per watt
    """

    VALUE = "%"
    BTU_IT_HP_H = "Btu[IT]/(hp.h)"
    EUC = "Euc"
    W_K_W = "W/kW"
    W_W = "W/W"


class PowerPerVolumeUom(Enum):
    """
    Properties
    ----------
    BTU_IT_H_FT3
        BTU per hour cubic foot
    BTU_IT_S_FT3
        (BTU per second) per cubic foot
    CAL_TH_H_CM3
        calorie per hour cubic centimetre
    CAL_TH_S_CM3
        calorie per second cubic centimetre
    HP_FT3
        horsepower per cubic foot
    K_W_M3
        kilowatt per cubic metre
    U_W_M3
        microwatt per cubic metre
    W_M3
        watt per cubic metre
    """

    BTU_IT_H_FT3 = "Btu[IT]/(h.ft3)"
    BTU_IT_S_FT3 = "Btu[IT]/(s.ft3)"
    CAL_TH_H_CM3 = "cal[th]/(h.cm3)"
    CAL_TH_S_CM3 = "cal[th]/(s.cm3)"
    HP_FT3 = "hp/ft3"
    K_W_M3 = "kW/m3"
    U_W_M3 = "uW/m3"
    W_M3 = "W/m3"


class PowerUom(Enum):
    """
    Properties
    ----------
    C_W
        centiwatt
    D_W
        deciwatt
    EW
        exawatt
    F_W
        femtowatt
    GW
        gigawatt
    HP
        horsepower
    HP_ELEC
        electric-horsepower
    HP_HYD
        hydraulic-horsepower
    HP_METRIC
        metric-horsepower
    K_W
        kilowatt
    MW
        megawatt
    M_W_1
        milliwatt
    N_W
        nanowatt
    P_W
        picowatt
    TON_REFRIG
        ton-refrigeration
    TW
        terawatt
    U_W
        microwatt
    W
        watt
    """

    C_W = "cW"
    D_W = "dW"
    EW = "EW"
    F_W = "fW"
    GW = "GW"
    HP = "hp"
    HP_ELEC = "hp[elec]"
    HP_HYD = "hp[hyd]"
    HP_METRIC = "hp[metric]"
    K_W = "kW"
    MW = "MW"
    M_W_1 = "mW"
    N_W = "nW"
    P_W = "pW"
    TON_REFRIG = "tonRefrig"
    TW = "TW"
    U_W = "uW"
    W = "W"


class PressurePerTimeUom(Enum):
    """
    Properties
    ----------
    ATM_H
        standard atmosphere per hour
    BAR_H
        bar per hour
    K_PA_H
        kilopascal per hour
    K_PA_MIN
        kilopascal per min
    MPA_H
        megapascal per hour
    PA_H
        pascal per hour
    PA_S
        pascal per second
    PSI_H
        psi per hour
    PSI_MIN
        psi per minute
    """

    ATM_H = "atm/h"
    BAR_H = "bar/h"
    K_PA_H = "kPa/h"
    K_PA_MIN = "kPa/min"
    MPA_H = "MPa/h"
    PA_H = "Pa/h"
    PA_S = "Pa/s"
    PSI_H = "psi/h"
    PSI_MIN = "psi/min"


class PressurePerVolumeUom(Enum):
    """
    Properties
    ----------
    PA_M3
        pascal per cubic metre
    PSI2_D_C_P_FT3
        psi squared day per centipoise cubic foot
    """

    PA_M3 = "Pa/m3"
    PSI2_D_C_P_FT3 = "psi2.d/(cP.ft3)"


class PressureSquaredPerForceTimePerAreaUom(Enum):
    """
    Properties
    ----------
    VALUE_0_001_K_PA2_C_P
        kilopascal squared per thousand centipoise
    BAR2_C_P
        bar squared per centipoise
    K_PA2_C_P
        kilopascal squared per centipoise
    PA2_PA_S
        pascal squared per pascal second
    PSI2_C_P
        psi squared per centipoise
    """

    VALUE_0_001_K_PA2_C_P = "0.001 kPa2/cP"
    BAR2_C_P = "bar2/cP"
    K_PA2_C_P = "kPa2/cP"
    PA2_PA_S = "Pa2/(Pa.s)"
    PSI2_C_P = "psi2/cP"


class PressureSquaredUom(Enum):
    """
    Properties
    ----------
    BAR2
        bar squared
    GPA2
        gigapascal squared
    K_PA2
        kilopascal squared
    KPSI2
        (thousand psi) squared
    PA2
        pascal squared
    PSI2
        psi squared
    """

    BAR2 = "bar2"
    GPA2 = "GPa2"
    K_PA2 = "kPa2"
    KPSI2 = "kpsi2"
    PA2 = "Pa2"
    PSI2 = "psi2"


class PressureTimePerVolumeUom(Enum):
    """
    Properties
    ----------
    PA_S_M3
        pascal second per cubic metre
    PSI_D_BBL
        psi day per barrel
    """

    PA_S_M3 = "Pa.s/m3"
    PSI_D_BBL = "psi.d/bbl"


class PressureUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_LBF_FT2
        pound-force per hundred square foot
    AT
        technical atmosphere
    ATM
        standard atmosphere
    BAR
        bar
    CM_H2_O_4DEG_C
        centimetre of water at 4 degree Celsius
    C_PA
        centipascal
    D_PA
        decipascal
    DYNE_CM2
        dyne per square centimetre
    EPA
        exapascal
    F_PA
        femtopascal
    GPA
        gigapascal
    HBAR
        hundred bar
    IN_H2_O_39DEG_F
        inch of water at 39.2 degree Fahrenheit
    IN_H2_O_60DEG_F
        inch of water at 60 degree Fahrenheit
    IN_HG_32DEG_F
        inch of mercury at 32 degree Fahrenheit
    IN_HG_60DEG_F
        inch of mercury at 60 degree Fahrenheit
    KGF_CM2
        thousand gram-force per square centimetre
    KGF_M2
        thousand gram-force per square metre
    KGF_MM2
        thousand gram-force per square millimetre
    K_N_M2
        kilonewton per square metre
    K_PA
        kilopascal
    KPSI
        thousand psi
    LBF_FT2
        pound-force per square foot
    MBAR
        thousandth of bar
    MM_HG_0DEG_C
        millimetres of Mercury at 0 deg C
    M_PA
        millipascal
    MPA_1
        megapascal
    MPSI
        million psi
    N_M2
        newton per square metre
    N_MM2
        newton per square millimetre
    N_PA
        nanopascal
    PA
        pascal
    P_PA
        picopascal
    PSI
        pound-force per square inch
    TONF_UK_FT2
        UK ton-force per square foot
    TONF_US_FT2
        US ton-force per square foot
    TONF_US_IN2
        US ton-force per square inch
    TORR
        torr
    TPA
        terapascal
    UBAR
        millionth of bar
    UM_HG_0DEG_C
        micrometre of mercury at 0 degree Celsius
    U_PA
        micropascal
    UPSI
        millionth of psi
    """

    VALUE_0_01_LBF_FT2 = "0.01 lbf/ft2"
    AT = "at"
    ATM = "atm"
    BAR = "bar"
    CM_H2_O_4DEG_C = "cmH2O[4degC]"
    C_PA = "cPa"
    D_PA = "dPa"
    DYNE_CM2 = "dyne/cm2"
    EPA = "EPa"
    F_PA = "fPa"
    GPA = "GPa"
    HBAR = "hbar"
    IN_H2_O_39DEG_F = "inH2O[39degF]"
    IN_H2_O_60DEG_F = "inH2O[60degF]"
    IN_HG_32DEG_F = "inHg[32degF]"
    IN_HG_60DEG_F = "inHg[60degF]"
    KGF_CM2 = "kgf/cm2"
    KGF_M2 = "kgf/m2"
    KGF_MM2 = "kgf/mm2"
    K_N_M2 = "kN/m2"
    K_PA = "kPa"
    KPSI = "kpsi"
    LBF_FT2 = "lbf/ft2"
    MBAR = "mbar"
    MM_HG_0DEG_C = "mmHg[0degC]"
    M_PA = "mPa"
    MPA_1 = "MPa"
    MPSI = "Mpsi"
    N_M2 = "N/m2"
    N_MM2 = "N/mm2"
    N_PA = "nPa"
    PA = "Pa"
    P_PA = "pPa"
    PSI = "psi"
    TONF_UK_FT2 = "tonf[UK]/ft2"
    TONF_US_FT2 = "tonf[US]/ft2"
    TONF_US_IN2 = "tonf[US]/in2"
    TORR = "torr"
    TPA = "TPa"
    UBAR = "ubar"
    UM_HG_0DEG_C = "umHg[0degC]"
    U_PA = "uPa"
    UPSI = "upsi"


class QuantityOfLightUom(Enum):
    """
    Properties
    ----------
    LM_S
        lumen second
    """

    LM_S = "lm.s"


class RadianceUom(Enum):
    """
    Properties
    ----------
    W_M2_SR
        watt per square metre steradian
    """

    W_M2_SR = "W/(m2.sr)"


class RadiantIntensityUom(Enum):
    """
    Properties
    ----------
    W_SR
        watt per steradian
    """

    W_SR = "W/sr"


class ReciprocalAreaUom(Enum):
    """
    Properties
    ----------
    VALUE_1_FT2
        per square foot
    VALUE_1_KM2
        per square kilometre
    VALUE_1_M2
        per square metre
    VALUE_1_MI2
        per square mile
    """

    VALUE_1_FT2 = "1/ft2"
    VALUE_1_KM2 = "1/km2"
    VALUE_1_M2 = "1/m2"
    VALUE_1_MI2 = "1/mi2"


class ReciprocalElectricPotentialDifferenceUom(Enum):
    """
    Properties
    ----------
    VALUE_1_U_V
        per microvolt
    VALUE_1_V
        per volt
    """

    VALUE_1_U_V = "1/uV"
    VALUE_1_V = "1/V"


class ReciprocalForceUom(Enum):
    """
    Properties
    ----------
    VALUE_1_LBF
        per pound-force
    VALUE_1_N
        per Newton
    """

    VALUE_1_LBF = "1/lbf"
    VALUE_1_N = "1/N"


class ReciprocalLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_1_ANGSTROM
        per angstrom
    VALUE_1_CM
        per centimetre
    VALUE_1_FT
        per foot
    VALUE_1_IN
        per inch
    VALUE_1_M
        per metre
    VALUE_1_MI
        per mile
    VALUE_1_MM
        per millimetre
    VALUE_1_NM
        per nanometre
    VALUE_1_YD
        per yard
    VALUE_1_E_9_1_FT
        per thousand million foot
    """

    VALUE_1_ANGSTROM = "1/angstrom"
    VALUE_1_CM = "1/cm"
    VALUE_1_FT = "1/ft"
    VALUE_1_IN = "1/in"
    VALUE_1_M = "1/m"
    VALUE_1_MI = "1/mi"
    VALUE_1_MM = "1/mm"
    VALUE_1_NM = "1/nm"
    VALUE_1_YD = "1/yd"
    VALUE_1_E_9_1_FT = "1E-9 1/ft"


class ReciprocalMassTimeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_KG_S
        per (kilogram per second)
    BQ_KG
        becquerel per kilogram
    P_CI_G
        picocurie per gram
    """

    VALUE_1_KG_S = "1/(kg.s)"
    BQ_KG = "Bq/kg"
    P_CI_G = "pCi/g"


class ReciprocalMassUom(Enum):
    """
    Properties
    ----------
    VALUE_1_G
        per gram
    VALUE_1_KG
        per kilogram
    VALUE_1_LBM
        per pound
    """

    VALUE_1_G = "1/g"
    VALUE_1_KG = "1/kg"
    VALUE_1_LBM = "1/lbm"


class ReciprocalPressureUom(Enum):
    """
    Properties
    ----------
    VALUE_1_BAR
        per bar
    VALUE_1_K_PA
        per kilopascal
    VALUE_1_PA
        per pascal
    VALUE_1_P_PA
        per picopascal
    VALUE_1_PSI
        per psi
    VALUE_1_UPSI
        per millionth of psi
    """

    VALUE_1_BAR = "1/bar"
    VALUE_1_K_PA = "1/kPa"
    VALUE_1_PA = "1/Pa"
    VALUE_1_P_PA = "1/pPa"
    VALUE_1_PSI = "1/psi"
    VALUE_1_UPSI = "1/upsi"


class ReciprocalTimeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_A
        per julian-year
    VALUE_1_D
        per day
    VALUE_1_H
        per hour
    VALUE_1_MIN
        per minute
    VALUE_1_MS
        per millisecond
    VALUE_1_S
        per second
    VALUE_1_US
        per microsecond
    VALUE_1_WK
        per week
    """

    VALUE_1_A = "1/a"
    VALUE_1_D = "1/d"
    VALUE_1_H = "1/h"
    VALUE_1_MIN = "1/min"
    VALUE_1_MS = "1/ms"
    VALUE_1_S = "1/s"
    VALUE_1_US = "1/us"
    VALUE_1_WK = "1/wk"


class ReciprocalVolumeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_BBL
        per barrel
    VALUE_1_FT3
        per cubic foot
    VALUE_1_GAL_UK
        per UK gallon
    VALUE_1_GAL_US
        per US gallon
    VALUE_1_L
        per litre
    VALUE_1_M3
        per cubic metre
    """

    VALUE_1_BBL = "1/bbl"
    VALUE_1_FT3 = "1/ft3"
    VALUE_1_GAL_UK = "1/gal[UK]"
    VALUE_1_GAL_US = "1/gal[US]"
    VALUE_1_L = "1/L"
    VALUE_1_M3 = "1/m3"


class ReluctanceUom(Enum):
    """
    Properties
    ----------
    VALUE_1_H
        per henry
    """

    VALUE_1_H = "1/H"


class SecondMomentOfAreaUom(Enum):
    """
    Properties
    ----------
    CM4
        centimetre to the fourth power
    IN4
        inch to the fourth power
    M4
        metre to the fourth power
    """

    CM4 = "cm4"
    IN4 = "in4"
    M4 = "m4"


class SignalingEventPerTimeUom(Enum):
    """
    Properties
    ----------
    BD
        baud
    """

    BD = "Bd"


class SolidAngleUom(Enum):
    """
    Properties
    ----------
    SR
        steradian
    """

    SR = "sr"


class SpecificHeatCapacityUom(Enum):
    """
    Properties
    ----------
    BTU_IT_LBM_DELTA_F
        BTU per pound-mass delta Fahrenheit
    BTU_IT_LBM_DELTA_R
        BTU per pound-mass delta Rankine
    CAL_TH_G_DELTA_K
        calorie per gram delta kelvin
    J_G_DELTA_K
        joule per gram delta kelvin
    J_KG_DELTA_K
        joule per kilogram delta kelvin
    KCAL_TH_KG_DELTA_C
        thousand calorie per kilogram delta Celsius
    K_J_KG_DELTA_K
        kilojoule per kilogram delta kelvin
    K_W_H_KG_DELTA_C
        kilowatt hour per kilogram delta Celsius
    """

    BTU_IT_LBM_DELTA_F = "Btu[IT]/(lbm.deltaF)"
    BTU_IT_LBM_DELTA_R = "Btu[IT]/(lbm.deltaR)"
    CAL_TH_G_DELTA_K = "cal[th]/(g.deltaK)"
    J_G_DELTA_K = "J/(g.deltaK)"
    J_KG_DELTA_K = "J/(kg.deltaK)"
    KCAL_TH_KG_DELTA_C = "kcal[th]/(kg.deltaC)"
    K_J_KG_DELTA_K = "kJ/(kg.deltaK)"
    K_W_H_KG_DELTA_C = "kW.h/(kg.deltaC)"


class TemperatureIntervalPerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_DELTA_F_FT
        delta Fahrenheit per hundred foot
    DELTA_C_FT
        delta Celsius per foot
    DELTA_C_HM
        delta Celsius per hectometre
    DELTA_C_KM
        delta Celsius per kilometre
    DELTA_C_M
        delta Celsius per metre
    DELTA_F_FT
        delta Fahrenheit per foot
    DELTA_F_M
        delta Fahrenheit per metre
    DELTA_K_KM
        delta kelvin per kilometre
    DELTA_K_M
        delta kelvin per metre
    """

    VALUE_0_01_DELTA_F_FT = "0.01 deltaF/ft"
    DELTA_C_FT = "deltaC/ft"
    DELTA_C_HM = "deltaC/hm"
    DELTA_C_KM = "deltaC/km"
    DELTA_C_M = "deltaC/m"
    DELTA_F_FT = "deltaF/ft"
    DELTA_F_M = "deltaF/m"
    DELTA_K_KM = "deltaK/km"
    DELTA_K_M = "deltaK/m"


class TemperatureIntervalPerPressureUom(Enum):
    """
    Properties
    ----------
    DELTA_C_K_PA
        delta Celsius per kilopascal
    DELTA_F_PSI
        delta Fahrenheit per psi
    DELTA_K_PA
        delta kelvin per Pascal
    """

    DELTA_C_K_PA = "deltaC/kPa"
    DELTA_F_PSI = "deltaF/psi"
    DELTA_K_PA = "deltaK/Pa"


class TemperatureIntervalPerTimeUom(Enum):
    """
    Properties
    ----------
    DELTA_C_H
        delta Celsius per hour
    DELTA_C_MIN
        delta Celsius per minute
    DELTA_C_S
        delta Celsius per second
    DELTA_F_H
        delta Fahrenheit per hour
    DELTA_F_MIN
        delta Fahrenheit per minute
    DELTA_F_S
        delta Fahrenheit per second
    DELTA_K_S
        delta kelvin per second
    """

    DELTA_C_H = "deltaC/h"
    DELTA_C_MIN = "deltaC/min"
    DELTA_C_S = "deltaC/s"
    DELTA_F_H = "deltaF/h"
    DELTA_F_MIN = "deltaF/min"
    DELTA_F_S = "deltaF/s"
    DELTA_K_S = "deltaK/s"


class TemperatureIntervalUom(Enum):
    """
    Properties
    ----------
    DELTA_C
        delta Celsius
    DELTA_F
        delta Fahrenheit
    DELTA_K
        delta kelvin
    DELTA_R
        delta Rankine
    """

    DELTA_C = "deltaC"
    DELTA_F = "deltaF"
    DELTA_K = "deltaK"
    DELTA_R = "deltaR"


class ThermalConductanceUom(Enum):
    """
    Properties
    ----------
    W_DELTA_K
        watt per delta kelvin
    """

    W_DELTA_K = "W/deltaK"


class ThermalConductivityUom(Enum):
    """
    Properties
    ----------
    BTU_IT_H_FT_DELTA_F
        BTU per hour foot delta Fahrenheit
    CAL_TH_H_CM_DELTA_C
        calorie per hour centimetre delta Celsius
    CAL_TH_S_CM_DELTA_C
        calorie per second centimetre delta Celsius
    KCAL_TH_H_M_DELTA_C
        thousand calorie per hour metre delta Celsius
    W_M_DELTA_K
        watt per metre delta kelvin
    """

    BTU_IT_H_FT_DELTA_F = "Btu[IT]/(h.ft.deltaF)"
    CAL_TH_H_CM_DELTA_C = "cal[th]/(h.cm.deltaC)"
    CAL_TH_S_CM_DELTA_C = "cal[th]/(s.cm.deltaC)"
    KCAL_TH_H_M_DELTA_C = "kcal[th]/(h.m.deltaC)"
    W_M_DELTA_K = "W/(m.deltaK)"


class ThermalDiffusivityUom(Enum):
    """
    Properties
    ----------
    CM2_S
        square centimetre per second
    FT2_H
        square foot per hour
    FT2_S
        square foot per second
    IN2_S
        square inch per second
    M2_H
        square metre per hour
    M2_S
        square metre per second
    MM2_S
        square millimetre per second
    """

    CM2_S = "cm2/s"
    FT2_H = "ft2/h"
    FT2_S = "ft2/s"
    IN2_S = "in2/s"
    M2_H = "m2/h"
    M2_S = "m2/s"
    MM2_S = "mm2/s"


class ThermalInsulanceUom(Enum):
    """
    Properties
    ----------
    DELTA_C_M2_H_KCAL_TH
        delta Celsius square metre hour per thousand calory
    DELTA_F_FT2_H_BTU_IT
        delta Fahrenheit square foot hour per BTU
    DELTA_K_M2_K_W
        delta kelvin square metre per kilowatt
    DELTA_K_M2_W
        delta kelvin square metre per watt
    """

    DELTA_C_M2_H_KCAL_TH = "deltaC.m2.h/kcal[th]"
    DELTA_F_FT2_H_BTU_IT = "deltaF.ft2.h/Btu[IT]"
    DELTA_K_M2_K_W = "deltaK.m2/kW"
    DELTA_K_M2_W = "deltaK.m2/W"


class ThermalResistanceUom(Enum):
    """
    Properties
    ----------
    DELTA_K_W
        delta kelvin per watt
    """

    DELTA_K_W = "deltaK/W"


class ThermodynamicTemperatureUom(Enum):
    """
    Properties
    ----------
    DEG_C
        degree Celsius
    DEG_F
        degree Fahrenheit
    DEG_R
        degree Rankine
    K
        degree kelvin
    """

    DEG_C = "degC"
    DEG_F = "degF"
    DEG_R = "degR"
    K = "K"


class TimePerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_0_001_H_FT
        hour per thousand foot
    H_KM
        hour per kilometre
    MIN_FT
        minute per foot
    MIN_M
        minute per metre
    MS_CM
        millisecond per centimetre
    MS_FT
        millisecond per foot
    MS_IN
        millisecond per inch
    MS_M
        millisecond per metre
    NS_FT
        nanosecond per foot
    NS_M
        nanosecond per metre
    S_CM
        second per centimetre
    S_FT
        second per foot
    S_IN
        second per inch
    S_M
        second per metre
    US_FT
        microsecond per foot
    US_IN
        microsecond per inch
    US_M
        microsecond per metre
    """

    VALUE_0_001_H_FT = "0.001 h/ft"
    H_KM = "h/km"
    MIN_FT = "min/ft"
    MIN_M = "min/m"
    MS_CM = "ms/cm"
    MS_FT = "ms/ft"
    MS_IN = "ms/in"
    MS_M = "ms/m"
    NS_FT = "ns/ft"
    NS_M = "ns/m"
    S_CM = "s/cm"
    S_FT = "s/ft"
    S_IN = "s/in"
    S_M = "s/m"
    US_FT = "us/ft"
    US_IN = "us/in"
    US_M = "us/m"


class TimePerMassUom(Enum):
    """
    Properties
    ----------
    S_KG
        second per kilogram
    """

    S_KG = "s/kg"


class TimePerTimeUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    EUC
        euclid
    MS_S
        millisecond per second
    S_S
        second per second
    """

    VALUE = "%"
    EUC = "Euc"
    MS_S = "ms/s"
    S_S = "s/s"


class TimePerVolumeUom(Enum):
    """
    Properties
    ----------
    VALUE_0_001_D_FT3
        day per thousand cubic foot
    D_BBL
        day per barrel
    D_FT3
        day per cubic foot
    D_M3
        day per cubic metre
    H_FT3
        hour per cubic foot
    H_M3
        hour per cubic metre
    S_FT3
        second per cubic foot
    S_L
        second per litre
    S_M3
        second per cubic metre
    S_QT_UK
        second per UK quart
    S_QT_US
        second per US quart
    """

    VALUE_0_001_D_FT3 = "0.001 d/ft3"
    D_BBL = "d/bbl"
    D_FT3 = "d/ft3"
    D_M3 = "d/m3"
    H_FT3 = "h/ft3"
    H_M3 = "h/m3"
    S_FT3 = "s/ft3"
    S_L = "s/L"
    S_M3 = "s/m3"
    S_QT_UK = "s/qt[UK]"
    S_QT_US = "s/qt[US]"


class TimeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_2_MS
        half of millisecond
    VALUE_100_KA_T
        hundred thousand tropical-year
    A
        julian-year
    A_T
        tropical-year
    CA
        hundredth of julian-year
    CS
        centisecond
    D
        day
    DS
        decisecond
    EA_T
        million million million tropical-year
    FA
        femtojulian-year
    GA_T
        thousand million tropical-year
    H
        hour
    HS
        hectosecond
    KA_T
        thousand tropical-year
    MA_T
        million tropical-year
    MIN
        minute
    MS
        millisecond
    NA
        nanojulian-year
    NS
        nanosecond
    PS
        picosecond
    S
        second
    TA_T
        million million tropical-year
    US
        microsecond
    WK
        week
    """

    VALUE_1_2_MS = "1/2 ms"
    VALUE_100_KA_T = "100 ka[t]"
    A = "a"
    A_T = "a[t]"
    CA = "ca"
    CS = "cs"
    D = "d"
    DS = "ds"
    EA_T = "Ea[t]"
    FA = "fa"
    GA_T = "Ga[t]"
    H = "h"
    HS = "hs"
    KA_T = "ka[t]"
    MA_T = "Ma[t]"
    MIN = "min"
    MS = "ms"
    NA = "na"
    NS = "ns"
    PS = "ps"
    S = "s"
    TA_T = "Ta[t]"
    US = "us"
    WK = "wk"


class VerticalDirection(Enum):
    """
    Properties
    ----------
    UP
        Values are positive when moving away from the center of the Earth.
    DOWN
        Values are positive when moving toward the center of the Earth.
    """

    UP = "up"
    DOWN = "down"


class VolumeFlowRatePerVolumeFlowRateUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    BBL_D_BBL_D
        (barrel per day) per (barrel per day)
    M3_D_M3_D
        (cubic metre per day) per (cubic metre per day)
    M3_S_M3_S
        (cubic metre per second) per (cubic metre per second)
    VALUE_1_E6_FT3_D_BBL_D
        (million cubic foot per day) per (barrel per day)
    EUC
        euclid
    """

    VALUE = "%"
    BBL_D_BBL_D = "(bbl/d)/(bbl/d)"
    M3_D_M3_D = "(m3/d)/(m3/d)"
    M3_S_M3_S = "(m3/s)/(m3/s)"
    VALUE_1_E6_FT3_D_BBL_D = "1E6 (ft3/d)/(bbl/d)"
    EUC = "Euc"


class VolumePerAreaUom(Enum):
    """
    Properties
    ----------
    VALUE_1_E6_BBL_ACRE
        million barrel per acre
    BBL_ACRE
        barrel per acre
    FT3_FT2
        cubic foot per square foot
    M3_M2
        cubic metre per square metre
    """

    VALUE_1_E6_BBL_ACRE = "1E6 bbl/acre"
    BBL_ACRE = "bbl/acre"
    FT3_FT2 = "ft3/ft2"
    M3_M2 = "m3/m2"


class VolumePerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_DM3_KM
        cubic decimetre per hundred kilometre
    VALUE_0_01_L_KM
        litre per hundred kilometre
    BBL_FT
        barrel per foot
    BBL_IN
        barrel per inch
    BBL_MI
        barrel per mile
    DM3_M
        cubic decimetre per metre
    FT3_FT
        cubic foot per foot
    GAL_UK_MI
        UK gallon per mile
    GAL_US_FT
        US gallon per foot
    GAL_US_MI
        US gallon per mile
    IN3_FT
        cubic inch per foot
    L_M
        litre per metre
    M3_KM
        cubic metre per kilometre
    M3_M
        cubic metre per metre
    """

    VALUE_0_01_DM3_KM = "0.01 dm3/km"
    VALUE_0_01_L_KM = "0.01 L/km"
    BBL_FT = "bbl/ft"
    BBL_IN = "bbl/in"
    BBL_MI = "bbl/mi"
    DM3_M = "dm3/m"
    FT3_FT = "ft3/ft"
    GAL_UK_MI = "gal[UK]/mi"
    GAL_US_FT = "gal[US]/ft"
    GAL_US_MI = "gal[US]/mi"
    IN3_FT = "in3/ft"
    L_M = "L/m"
    M3_KM = "m3/km"
    M3_M = "m3/m"


class VolumePerMassUom(Enum):
    """
    Properties
    ----------
    VALUE_0_01_L_KG
        litre per hundred kilogram
    BBL_TON_UK
        barrel per UK ton-mass
    BBL_TON_US
        barrel per US ton-mass
    CM3_G
        cubic centimetre per gram
    DM3_KG
        cubic decimetre per kilogram
    DM3_T
        cubic decimetre per ton
    FT3_KG
        cubic foot per kilogram
    FT3_LBM
        cubic foot per pound-mass
    FT3_SACK_94LBM
        cubic foot per 94-pound-sack
    GAL_UK_LBM
        UK gallon per pound-mass
    GAL_US_LBM
        US gallon per pound-mass
    GAL_US_SACK_94LBM
        US gallon per 94-pound-sack
    GAL_US_TON_UK
        US gallon per UK ton-mass
    GAL_US_TON_US
        US gallon per US ton-mass
    L_KG
        litre per kilogram
    L_T
        litre per tonne
    L_TON_UK
        litre per UK ton-mass
    M3_G
        cubic metre per gram
    M3_KG
        cubic metre per kilogram
    M3_T
        cubic metre per tonne
    M3_TON_UK
        cubic metre per UK ton-mass
    M3_TON_US
        cubic metre per US ton-mass
    """

    VALUE_0_01_L_KG = "0.01 L/kg"
    BBL_TON_UK = "bbl/ton[UK]"
    BBL_TON_US = "bbl/ton[US]"
    CM3_G = "cm3/g"
    DM3_KG = "dm3/kg"
    DM3_T = "dm3/t"
    FT3_KG = "ft3/kg"
    FT3_LBM = "ft3/lbm"
    FT3_SACK_94LBM = "ft3/sack[94lbm]"
    GAL_UK_LBM = "gal[UK]/lbm"
    GAL_US_LBM = "gal[US]/lbm"
    GAL_US_SACK_94LBM = "gal[US]/sack[94lbm]"
    GAL_US_TON_UK = "gal[US]/ton[UK]"
    GAL_US_TON_US = "gal[US]/ton[US]"
    L_KG = "L/kg"
    L_T = "L/t"
    L_TON_UK = "L/ton[UK]"
    M3_G = "m3/g"
    M3_KG = "m3/kg"
    M3_T = "m3/t"
    M3_TON_UK = "m3/ton[UK]"
    M3_TON_US = "m3/ton[US]"


class VolumePerPressureUom(Enum):
    """
    Properties
    ----------
    BBL_PSI
        barrel per psi
    M3_K_PA
        cubic metre per kilopascal
    M3_PA
        cubic metre per Pascal
    """

    BBL_PSI = "bbl/psi"
    M3_K_PA = "m3/kPa"
    M3_PA = "m3/Pa"


class VolumePerRotationUom(Enum):
    """
    Properties
    ----------
    FT3_RAD
        cubic foot per radian
    M3_RAD
        cubic metre per radian
    M3_REV
        cubic metre per revolution
    """

    FT3_RAD = "ft3/rad"
    M3_RAD = "m3/rad"
    M3_REV = "m3/rev"


class VolumePerTimeLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_1000_BBL_FT_D
        thousand barrel foot per day
    VALUE_1000_M4_D
        thousand (cubic metre per day) metre
    M4_S
        metre to the fourth power per second
    """

    VALUE_1000_BBL_FT_D = "1000 bbl.ft/d"
    VALUE_1000_M4_D = "1000 m4/d"
    M4_S = "m4/s"


class VolumePerTimePerAreaUom(Enum):
    """
    Properties
    ----------
    FT3_MIN_FT2
        cubic foot per minute square foot
    FT3_S_FT2
        cubic foot per second square foot
    GAL_UK_H_FT2
        UK gallon per hour square foot
    GAL_UK_H_IN2
        UK gallon per hour square inch
    GAL_UK_MIN_FT2
        UK gallon per minute square foot
    GAL_US_H_FT2
        US gallon per hour square foot
    GAL_US_H_IN2
        US gallon per hour square inch
    GAL_US_MIN_FT2
        US gallon per minute square foot
    M3_S_M2
        cubic metre per second square metre
    """

    FT3_MIN_FT2 = "ft3/(min.ft2)"
    FT3_S_FT2 = "ft3/(s.ft2)"
    GAL_UK_H_FT2 = "gal[UK]/(h.ft2)"
    GAL_UK_H_IN2 = "gal[UK]/(h.in2)"
    GAL_UK_MIN_FT2 = "gal[UK]/(min.ft2)"
    GAL_US_H_FT2 = "gal[US]/(h.ft2)"
    GAL_US_H_IN2 = "gal[US]/(h.in2)"
    GAL_US_MIN_FT2 = "gal[US]/(min.ft2)"
    M3_S_M2 = "m3/(s.m2)"


class VolumePerTimePerLengthUom(Enum):
    """
    Properties
    ----------
    VALUE_1000_FT3_D_FT
        (thousand cubic foot per day) per foot
    VALUE_1000_M3_D_M
        (thousand cubic metre per day) per metre
    VALUE_1000_M3_H_M
        (thousand cubic metre per hour) per metre
    BBL_D_FT
        barrel per day foot
    FT3_D_FT
        (cubic foot per day) per foot
    GAL_UK_H_FT
        UK gallon per hour foot
    GAL_UK_H_IN
        UK gallon per hour inch
    GAL_UK_MIN_FT
        UK gallon per minute foot
    GAL_US_H_FT
        US gallon per hour foot
    GAL_US_H_IN
        US gallon per hour inch
    GAL_US_MIN_FT
        US gallon per minute foot
    M3_D_M
        (cubic metre per day) per metre
    M3_H_M
        (cubic metre per hour) per metre
    M3_S_FT
        (cubic metre per second) per foot
    M3_S_M
        cubic metre per second metre
    """

    VALUE_1000_FT3_D_FT = "1000 ft3/(d.ft)"
    VALUE_1000_M3_D_M = "1000 m3/(d.m)"
    VALUE_1000_M3_H_M = "1000 m3/(h.m)"
    BBL_D_FT = "bbl/(d.ft)"
    FT3_D_FT = "ft3/(d.ft)"
    GAL_UK_H_FT = "gal[UK]/(h.ft)"
    GAL_UK_H_IN = "gal[UK]/(h.in)"
    GAL_UK_MIN_FT = "gal[UK]/(min.ft)"
    GAL_US_H_FT = "gal[US]/(h.ft)"
    GAL_US_H_IN = "gal[US]/(h.in)"
    GAL_US_MIN_FT = "gal[US]/(min.ft)"
    M3_D_M = "m3/(d.m)"
    M3_H_M = "m3/(h.m)"
    M3_S_FT = "m3/(s.ft)"
    M3_S_M = "m3/(s.m)"


class VolumePerTimePerPressureLengthUom(Enum):
    """
    Properties
    ----------
    BBL_FT_PSI_D
        barrel per day foot psi
    FT3_FT_PSI_D
        cubic foot per day foot psi
    M2_K_PA_D
        square metre per kilopascal day
    M2_PA_S
        square metre per pascal second
    """

    BBL_FT_PSI_D = "bbl/(ft.psi.d)"
    FT3_FT_PSI_D = "ft3/(ft.psi.d)"
    M2_K_PA_D = "m2/(kPa.d)"
    M2_PA_S = "m2/(Pa.s)"


class VolumePerTimePerPressureUom(Enum):
    """
    Properties
    ----------
    VALUE_1000_FT3_PSI_D
        (thousand cubic foot per day) per psi
    BBL_K_PA_D
        (barrel per day) per kilopascal
    BBL_PSI_D
        (barrel per day) per psi
    L_BAR_MIN
        (litre per minute) per bar
    M3_BAR_D
        (cubic metre per day) per bar
    M3_BAR_H
        (cubic metre per hour) per bar
    M3_BAR_MIN
        (cubic metre per minute) per bar
    M3_K_PA_D
        (cubic metre per day) per kilopascal
    M3_K_PA_H
        (cubic metre per hour) per kilopascal
    M3_PA_S
        cubic metre per pascal second
    M3_PSI_D
        (cubic metre per day) per psi
    """

    VALUE_1000_FT3_PSI_D = "1000 ft3/(psi.d)"
    BBL_K_PA_D = "bbl/(kPa.d)"
    BBL_PSI_D = "bbl/(psi.d)"
    L_BAR_MIN = "L/(bar.min)"
    M3_BAR_D = "m3/(bar.d)"
    M3_BAR_H = "m3/(bar.h)"
    M3_BAR_MIN = "m3/(bar.min)"
    M3_K_PA_D = "m3/(kPa.d)"
    M3_K_PA_H = "m3/(kPa.h)"
    M3_PA_S = "m3/(Pa.s)"
    M3_PSI_D = "m3/(psi.d)"


class VolumePerTimePerTimeUom(Enum):
    """
    Properties
    ----------
    BBL_D2
        (barrel per day) per day
    BBL_H2
        (barrel per hour) per hour
    DM3_S2
        (cubic decimetre per second) per second
    FT3_D2
        (cubic foot per day) per day
    FT3_H2
        (cubic foot per hour) per hour
    FT3_MIN2
        (cubic foot per minute) per minute
    FT3_S2
        (cubic foot per second) per second
    GAL_UK_H2
        (UK gallon per hour) per hour
    GAL_UK_MIN2
        (UK gallon per minute) per minute
    GAL_US_H2
        (US gallon per hour) per hour
    GAL_US_MIN2
        (US gallon per minute) per minute
    L_S2
        (litre per second) per second
    M3_D2
        (cubic metre per day) per day
    M3_S2
        cubic metre per second squared
    """

    BBL_D2 = "bbl/d2"
    BBL_H2 = "bbl/h2"
    DM3_S2 = "dm3/s2"
    FT3_D2 = "ft3/d2"
    FT3_H2 = "ft3/h2"
    FT3_MIN2 = "ft3/min2"
    FT3_S2 = "ft3/s2"
    GAL_UK_H2 = "gal[UK]/h2"
    GAL_UK_MIN2 = "gal[UK]/min2"
    GAL_US_H2 = "gal[US]/h2"
    GAL_US_MIN2 = "gal[US]/min2"
    L_S2 = "L/s2"
    M3_D2 = "m3/d2"
    M3_S2 = "m3/s2"


class VolumePerTimePerVolumeUom(Enum):
    """
    Properties
    ----------
    BBL_D_ACRE_FT
        barrel per day acre foot
    M3_S_M3
        cubic metre per time cubic metre
    """

    BBL_D_ACRE_FT = "bbl/(d.acre.ft)"
    M3_S_M3 = "m3/(s.m3)"


class VolumePerTimeUom(Enum):
    """
    Properties
    ----------
    VALUE_1_30_CM3_MIN
        cubic centimetre per thirty minute
    VALUE_1000_BBL_D
        thousand barrel per day
    VALUE_1000_FT3_D
        thousand cubic foot per day
    VALUE_1000_M3_D
        thousand cubic metre per day
    VALUE_1000_M3_H
        thousand cubic metre per hour
    VALUE_1_E6_BBL_D
        million barrel per day
    VALUE_1_E6_FT3_D
        million cubic foot per day
    VALUE_1_E6_M3_D
        million cubic metre per day
    BBL_D
        barrel per day
    BBL_H
        barrel per hour
    BBL_MIN
        barrel per minute
    CM3_H
        cubic centimetre per hour
    CM3_MIN
        cubic centimetre per minute
    CM3_S
        cubic centimetre per second
    DM3_S
        cubic decimetre per second
    FT3_D
        cubic foot per day
    FT3_H
        cubic foot per hour
    FT3_MIN
        cubic foot per minute
    FT3_S
        cubic foot per second
    GAL_UK_D
        UK gallon per day
    GAL_UK_H
        UK gallon per hour
    GAL_UK_MIN
        UK gallon per minute
    GAL_US_D
        US gallon per day
    GAL_US_H
        US gallon per hour
    GAL_US_MIN
        US gallon per minute
    L_H
        litre per hour
    L_MIN
        litre per minute
    L_S
        litre per second
    M3_D
        cubic metre per day
    M3_H
        cubic metre per hour
    M3_MIN
        cubic metre per minute
    M3_S
        cubic metre per second
    """

    VALUE_1_30_CM3_MIN = "1/30 cm3/min"
    VALUE_1000_BBL_D = "1000 bbl/d"
    VALUE_1000_FT3_D = "1000 ft3/d"
    VALUE_1000_M3_D = "1000 m3/d"
    VALUE_1000_M3_H = "1000 m3/h"
    VALUE_1_E6_BBL_D = "1E6 bbl/d"
    VALUE_1_E6_FT3_D = "1E6 ft3/d"
    VALUE_1_E6_M3_D = "1E6 m3/d"
    BBL_D = "bbl/d"
    BBL_H = "bbl/h"
    BBL_MIN = "bbl/min"
    CM3_H = "cm3/h"
    CM3_MIN = "cm3/min"
    CM3_S = "cm3/s"
    DM3_S = "dm3/s"
    FT3_D = "ft3/d"
    FT3_H = "ft3/h"
    FT3_MIN = "ft3/min"
    FT3_S = "ft3/s"
    GAL_UK_D = "gal[UK]/d"
    GAL_UK_H = "gal[UK]/h"
    GAL_UK_MIN = "gal[UK]/min"
    GAL_US_D = "gal[US]/d"
    GAL_US_H = "gal[US]/h"
    GAL_US_MIN = "gal[US]/min"
    L_H = "L/h"
    L_MIN = "L/min"
    L_S = "L/s"
    M3_D = "m3/d"
    M3_H = "m3/h"
    M3_MIN = "m3/min"
    M3_S = "m3/s"


class VolumePerVolumeUom(Enum):
    """
    Properties
    ----------
    VALUE
        percent
    VOL
        percent [volume basis]
    VALUE_0_001_BBL_FT3
        barrel per thousand cubic foot
    VALUE_0_001_BBL_M3
        barrel per thousand cubic metre
    VALUE_0_001_GAL_UK_BBL
        UK gallon per thousand barrel
    VALUE_0_001_GAL_UK_GAL_UK
        UK gallon per thousand UK gallon
    VALUE_0_001_GAL_US_BBL
        US gallon per thousand barrel
    VALUE_0_001_GAL_US_FT3
        US gallon per thousand cubic foot
    VALUE_0_001_GAL_US_GAL_US
        US gallon per thousand US gallon
    VALUE_0_001_PT_UK_BBL
        UK pint per thousand barrel
    VALUE_0_01_BBL_BBL
        barrel per hundred barrel
    VALUE_0_1_GAL_US_BBL
        US gallon per ten barrel
    VALUE_0_1_L_BBL
        litre per ten barrel
    VALUE_0_1_PT_US_BBL
        US pint per ten barrel
    VALUE_1000_FT3_BBL
        thousand cubic foot per barrel
    VALUE_1000_M3_M3
        thousand cubic metre per cubic metre
    VALUE_1_E_6_ACRE_FT_BBL
        acre foot per million barrel
    VALUE_1_E_6_BBL_FT3
        barrel per million cubic foot
    VALUE_1_E_6_BBL_M3
        barrel per million cubic metre
    VALUE_1_E6_BBL_ACRE_FT
        million barrel per acre foot
    VALUE_1_E6_FT3_ACRE_FT
        million cubic foot per acre foot
    VALUE_1_E6_FT3_BBL
        million cubic foot per barrel
    BBL_ACRE_FT
        barrel per acre foot
    BBL_BBL
        barrel per barrel
    BBL_FT3
        barrel per cubic foot
    BBL_M3
        barrel per cubic metre
    C_EUC
        centieuclid
    CM3_CM3
        cubic centimetre per cubic centimetre
    CM3_L
        cubic centimetre per litre
    CM3_M3
        cubic centimetre per cubic metre
    DM3_M3
        cubic decimetre per cubic metre
    EUC
        euclid
    FT3_BBL
        cubic foot per barrel
    FT3_FT3
        cubic foot per cubic foot
    GAL_UK_FT3
        UK gallon per cubic foot
    GAL_US_BBL
        US gallon per barrel
    GAL_US_FT3
        US gallon per cubic foot
    L_M3
        litre per cubic metre
    M3_HA_M
        cubic metre per hectare metre
    M3_BBL
        cubic metre per barrel
    M3_M3
        cubic metre per cubic metre
    M_L_GAL_UK
        millilitre per UK gallon
    M_L_GAL_US
        millilitre per US gallon
    M_L_M_L
        millilitre per millilitre
    PPK
        part per thousand
    PPM
        part per million
    PPM_VOL
        part per million [volume basis]
    """

    VALUE = "%"
    VOL = "%[vol]"
    VALUE_0_001_BBL_FT3 = "0.001 bbl/ft3"
    VALUE_0_001_BBL_M3 = "0.001 bbl/m3"
    VALUE_0_001_GAL_UK_BBL = "0.001 gal[UK]/bbl"
    VALUE_0_001_GAL_UK_GAL_UK = "0.001 gal[UK]/gal[UK]"
    VALUE_0_001_GAL_US_BBL = "0.001 gal[US]/bbl"
    VALUE_0_001_GAL_US_FT3 = "0.001 gal[US]/ft3"
    VALUE_0_001_GAL_US_GAL_US = "0.001 gal[US]/gal[US]"
    VALUE_0_001_PT_UK_BBL = "0.001 pt[UK]/bbl"
    VALUE_0_01_BBL_BBL = "0.01 bbl/bbl"
    VALUE_0_1_GAL_US_BBL = "0.1 gal[US]/bbl"
    VALUE_0_1_L_BBL = "0.1 L/bbl"
    VALUE_0_1_PT_US_BBL = "0.1 pt[US]/bbl"
    VALUE_1000_FT3_BBL = "1000 ft3/bbl"
    VALUE_1000_M3_M3 = "1000 m3/m3"
    VALUE_1_E_6_ACRE_FT_BBL = "1E-6 acre.ft/bbl"
    VALUE_1_E_6_BBL_FT3 = "1E-6 bbl/ft3"
    VALUE_1_E_6_BBL_M3 = "1E-6 bbl/m3"
    VALUE_1_E6_BBL_ACRE_FT = "1E6 bbl/(acre.ft)"
    VALUE_1_E6_FT3_ACRE_FT = "1E6 ft3/(acre.ft)"
    VALUE_1_E6_FT3_BBL = "1E6 ft3/bbl"
    BBL_ACRE_FT = "bbl/(acre.ft)"
    BBL_BBL = "bbl/bbl"
    BBL_FT3 = "bbl/ft3"
    BBL_M3 = "bbl/m3"
    C_EUC = "cEuc"
    CM3_CM3 = "cm3/cm3"
    CM3_L = "cm3/L"
    CM3_M3 = "cm3/m3"
    DM3_M3 = "dm3/m3"
    EUC = "Euc"
    FT3_BBL = "ft3/bbl"
    FT3_FT3 = "ft3/ft3"
    GAL_UK_FT3 = "gal[UK]/ft3"
    GAL_US_BBL = "gal[US]/bbl"
    GAL_US_FT3 = "gal[US]/ft3"
    L_M3 = "L/m3"
    M3_HA_M = "m3/(ha.m)"
    M3_BBL = "m3/bbl"
    M3_M3 = "m3/m3"
    M_L_GAL_UK = "mL/gal[UK]"
    M_L_GAL_US = "mL/gal[US]"
    M_L_M_L = "mL/mL"
    PPK = "ppk"
    PPM = "ppm"
    PPM_VOL = "ppm[vol]"


class VolumeUom(Enum):
    """
    Properties
    ----------
    VALUE_1000_BBL
        thousand barrel
    VALUE_1000_FT3
        thousand cubic foot
    VALUE_1000_GAL_UK
        thousand UK gallon
    VALUE_1000_GAL_US
        thousand US gallon
    VALUE_1000_M3
        thousand cubic metre
    VALUE_1_E_6_GAL_US
        millionth of US gallon
    VALUE_1_E12_FT3
        million million cubic foot
    VALUE_1_E6_BBL
        million barrel
    VALUE_1_E6_FT3
        million cubic foot
    VALUE_1_E6_M3
        million cubic metre
    VALUE_1_E9_BBL
        thousand million barrel
    VALUE_1_E9_FT3
        thousand million cubic foot
    ACRE_FT
        acre foot
    BBL
        barrel
    CM3
        cubic centimetre
    DM3
        cubic decimetre
    FLOZ_UK
        UK fluid-ounce
    FLOZ_US
        US fluid-ounce
    FT3
        cubic foot
    GAL_UK
        UK gallon
    GAL_US
        US gallon
    HA_M
        hectare metre
    H_L
        hectolitre
    IN3
        cubic inch
    KM3
        cubic kilometre
    L
        litre
    M3
        cubic metre
    MI3
        cubic mile
    M_L
        millilitre
    MM3
        cubic millimetre
    PT_UK
        UK pint
    PT_US
        US pint
    QT_UK
        UK quart
    QT_US
        US quart
    UM2_M
        square micrometre metre
    YD3
        cubic yard
    """

    VALUE_1000_BBL = "1000 bbl"
    VALUE_1000_FT3 = "1000 ft3"
    VALUE_1000_GAL_UK = "1000 gal[UK]"
    VALUE_1000_GAL_US = "1000 gal[US]"
    VALUE_1000_M3 = "1000 m3"
    VALUE_1_E_6_GAL_US = "1E-6 gal[US]"
    VALUE_1_E12_FT3 = "1E12 ft3"
    VALUE_1_E6_BBL = "1E6 bbl"
    VALUE_1_E6_FT3 = "1E6 ft3"
    VALUE_1_E6_M3 = "1E6 m3"
    VALUE_1_E9_BBL = "1E9 bbl"
    VALUE_1_E9_FT3 = "1E9 ft3"
    ACRE_FT = "acre.ft"
    BBL = "bbl"
    CM3 = "cm3"
    DM3 = "dm3"
    FLOZ_UK = "floz[UK]"
    FLOZ_US = "floz[US]"
    FT3 = "ft3"
    GAL_UK = "gal[UK]"
    GAL_US = "gal[US]"
    HA_M = "ha.m"
    H_L = "hL"
    IN3 = "in3"
    KM3 = "km3"
    L = "L"
    M3 = "m3"
    MI3 = "mi3"
    M_L = "mL"
    MM3 = "mm3"
    PT_UK = "pt[UK]"
    PT_US = "pt[US]"
    QT_UK = "qt[UK]"
    QT_US = "qt[US]"
    UM2_M = "um2.m"
    YD3 = "yd3"


class VolumetricHeatTransferCoefficientUom(Enum):
    """
    Properties
    ----------
    BTU_IT_H_FT3_DELTA_F
        BTU per hour cubic foot delta Fahrenheit
    BTU_IT_S_FT3_DELTA_F
        (BTU per second) per cubic foot delta Fahrenheit
    K_W_M3_DELTA_K
        killowatt per cubic metre delta kelvin
    W_M3_DELTA_K
        watt per cubic metre delta kelvin
    """

    BTU_IT_H_FT3_DELTA_F = "Btu[IT]/(h.ft3.deltaF)"
    BTU_IT_S_FT3_DELTA_F = "Btu[IT]/(s.ft3.deltaF)"
    K_W_M3_DELTA_K = "kW/(m3.deltaK)"
    W_M3_DELTA_K = "W/(m3.deltaK)"


class VolumetricThermalExpansionUom(Enum):
    """
    Properties
    ----------
    VALUE_1_DELTA_C
        per delta Celsius
    VALUE_1_DELTA_F
        per delta Fahrenheit
    VALUE_1_DELTA_K
        per delta kelvin
    VALUE_1_DELTA_R
        per delta Rankine
    VALUE_1_E_6_M3_M3_DELTA_C
        (cubic metre per million cubic metre) per delta Celsius
    VALUE_1_E_6_M3_M3_DELTA_F
        (cubic metre per million cubic metre) per delta Fahrenheit
    M3_M3_DELTA_K
        cubic metre per cubic metre delta kelvin
    PPM_VOL_DELTA_C
        (part per million [volume basis]) per delta Celsius
    PPM_VOL_DELTA_F
        (part per million [volume basis)] per delta Fahrenheit
    """

    VALUE_1_DELTA_C = "1/deltaC"
    VALUE_1_DELTA_F = "1/deltaF"
    VALUE_1_DELTA_K = "1/deltaK"
    VALUE_1_DELTA_R = "1/deltaR"
    VALUE_1_E_6_M3_M3_DELTA_C = "1E-6 m3/(m3.deltaC)"
    VALUE_1_E_6_M3_M3_DELTA_F = "1E-6 m3/(m3.deltaF)"
    M3_M3_DELTA_K = "m3/(m3.deltaK)"
    PPM_VOL_DELTA_C = "ppm[vol]/deltaC"
    PPM_VOL_DELTA_F = "ppm[vol]/deltaF"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractContactRepresentationPart:
    """
    Parent class of the sealed and nonsealed contact elements.

    Parameters
    ----------
    index
        The index of the contact. Indicates identity of the contact in the
        surface framework context. It is used for contact identities and to
        find the interpretation of this particular contact.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractParameterKey:
    """Abstract class describing a key used to identify a parameter value.

    When multiple values are provided for a given parameter, provides a way to identify the parameter through its association with an object, a time index...
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractParametricLineArray:
    """Defines an array of parametric lines.

    The array size is obtained from context. In the current schema, this
    may be as simple as a 1D array (#Lines=count) or a 2D array #Lines =
    NIL x NJL for an IJK grid representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractPoint3dArray:
    """The abstract class of 3D points implemented in a single fashion for the
    schema.

    Abstraction allows a variety of instantiations for efficiency or to
    implicitly provide additional geometric information about a data-
    object. For example, parametric points can be used to implicitly
    define a wellbore trajectory using an underlying parametric line, by
    the specification of the control points along the parametric line.
    The dimensionality of the array of 3D points is based on context
    within an instance.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractPropertyKind:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractValueArray:
    """Generic representation of an array of numeric, Boolean, and string values.

    Each derived element provides specialized implementation for
    specific content types or for optimization of the representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


class BoundaryRelation(Enum):
    """
    The enumerated attributes of a horizon.

    Properties
    ----------
    CONFORMABLE
        If used uniquely, then it means the horizon is conformable above and
        below. If used with unconformity, then it means partial
        unconformity.
    UNCONFORMABLE_BELOW_AND_ABOVE
    UNCONFORMABLE_ABOVE
        If used with conformable, then it means partial unconformity.
    UNCONFORMABLE_BELOW
        If used with conformable, then it means partial unconformity.
    """

    CONFORMABLE = "conformable"
    UNCONFORMABLE_BELOW_AND_ABOVE = "unconformable below and above"
    UNCONFORMABLE_ABOVE = "unconformable above"
    UNCONFORMABLE_BELOW = "unconformable below"


class CellShape(Enum):
    """Used to indicate that all cells are of a uniform topology, i.e., have the
    same number of nodes per cell.

    This information is supplied by the RESQML writer to indicate the complexity of the grid geometry, as an aide to the RESQML reader.
    If a specific cell shape is not appropriate, then use polyhedral.
    BUSINESS RULE: Should be consistent with the actual geometry of the grid.

    Properties
    ----------
    TETRAHEDRAL
        All grid cells are constrained to have only 4 nodes/cell with 4
        faces/cell, 3 nodes/face, 4 nodes/cell for all cells (degeneracy
        allowed).
    PYRAMIDAL
        All grid cells are constrained to have only 5 nodes/cell with 5
        faces/cell, with 1 quadrilateral face and 4 triangular faces.
    PRISM
        All grid cells are constrained to have 6 nodes/cell with 5
        faces/cell, with 3 quadrilateral faces and 2 non-adjacent triangular
        faces, as in a column layer grid with triangular columns.
    HEXAHEDRAL
        All grid cells are constrained to have 8 nodes/cell with 6
        faces/cell, 4 nodes/face, 8 nodes/cell for all cells (degeneracy
        allowed). Equivalent to IJK grid cells.
    POLYHEDRAL
        If the cell geometry is not of a more specific kind, use polyhedral.
    """

    TETRAHEDRAL = "tetrahedral"
    PYRAMIDAL = "pyramidal"
    PRISM = "prism"
    HEXAHEDRAL = "hexahedral"
    POLYHEDRAL = "polyhedral"


class ColumnShape(Enum):
    """Used to indicate that all columns are of a uniform topology, i.e., have the
    same number of faces per column.

    This information is supplied by the RESQML writer to indicate the complexity of the grid geometry, as an aide to the RESQML reader.
    If a specific column shape is not appropriate, then use polygonal.
    BUSINESS RULE: Should be consistent with the actual geometry of the grid.

    Properties
    ----------
    TRIANGULAR
        All grid columns have 3 sides.
    QUADRILATERAL
        All grid columns have 4 sides. Includes tartan and corner point
        grids.
    POLYGONAL
        At least one grid column is a polygon, N&gt;4.
    """

    TRIANGULAR = "triangular"
    QUADRILATERAL = "quadrilateral"
    POLYGONAL = "polygonal"


class ContactMode(Enum):
    BASELAP = "baselap"
    EROSION = "erosion"
    EXTENDED = "extended"
    PROPORTIONAL = "proportional"


class ContactRelationship(Enum):
    """
    The enumerations that specify the role of the contacts in a contact
    interpretation as described in the attributes below.

    Properties
    ----------
    FRONTIER_FEATURE_TO_FRONTIER_FEATURE
        A contact between two frontier features to close a volume of
        interest.
    GENETIC_BOUNDARY_TO_FRONTIER_FEATURE
        A linear contact between a genetic boundary interpretation and a
        frontier feature (horizon/frontier contact).
    GENETIC_BOUNDARY_TO_GENETIC_BOUNDARY
        A linear contact between two genetic boundary interpretations
        (horizon/horizon contact).
    GENETIC_BOUNDARY_TO_TECTONIC_BOUNDARY
        A linear contact between a genetic boundary interpretation and a
        tectonic boundary interpretation (horizon/fault contact).
    STRATIGRAPHIC_UNIT_TO_FRONTIER_FEATURE
        A surface contact between a stratigraphic unit interpretation and a
        frontier feature (contact closing a volume on a frontier feature
        that is a technical feature).
    STRATIGRAPHIC_UNIT_TO_STRATIGRAPHIC_UNIT
        A surface contact between two stratigraphic unit interpretations
        (unit/unit contact).
    TECTONIC_BOUNDARY_TO_FRONTIER_FEATURE
        A linear contact between a tectonic boundary interpretation and a
        frontier feature (fault/frontier contact).
    TECTONIC_BOUNDARY_TO_GENETIC_BOUNDARY
        A linear contact between a tectonic boundary interpretation and a
        genetic boundary interpretation (fault/horizon contact).
    TECTONIC_BOUNDARY_TO_TECTONIC_BOUNDARY
        A linear contact between two tectonic boundary interpretations
        (fault/fault contact).
    """

    FRONTIER_FEATURE_TO_FRONTIER_FEATURE = "frontier feature to frontier feature"
    GENETIC_BOUNDARY_TO_FRONTIER_FEATURE = "genetic boundary to frontier feature"
    GENETIC_BOUNDARY_TO_GENETIC_BOUNDARY = "genetic boundary to genetic boundary"
    GENETIC_BOUNDARY_TO_TECTONIC_BOUNDARY = "genetic boundary to tectonic boundary"
    STRATIGRAPHIC_UNIT_TO_FRONTIER_FEATURE = "stratigraphic unit to frontier feature"
    STRATIGRAPHIC_UNIT_TO_STRATIGRAPHIC_UNIT = (
        "stratigraphic unit to stratigraphic unit"
    )
    TECTONIC_BOUNDARY_TO_FRONTIER_FEATURE = "tectonic boundary to frontier feature"
    TECTONIC_BOUNDARY_TO_GENETIC_BOUNDARY = "tectonic boundary to genetic boundary"
    TECTONIC_BOUNDARY_TO_TECTONIC_BOUNDARY = "tectonic boundary to tectonic boundary"


class ContactSide(Enum):
    """Enumeration that specifies the location of the contacts, chosen from the
    attributes listed below.

    For example, if you specify contact between a horizon and a fault, you can specify if the contact is on the foot wall side or the hanging wall side of the fault, and if the fault is splitting both sides of a horizon or the older side only.
    From Wikipedia: http://en.wikipedia.org/wiki/Foot_wall
    CC-BY-SA-3.0-MIGRATED; GFDL-WITH-DISCLAIMERS
    Released under the GNU Free Documentation License.

    Properties
    ----------
    FOOTWALL
        The footwall side of the fault. See picture.
    HANGING_WALL
        The hanging wall side of the fault. See picture.
    NORTH
        For a vertical fault, specification of the north side.
    SOUTH
        For a vertical fault, specification of the south side.
    EAST
        For a vertical fault, specification of the east side.
    WEST
        For a vertical fault, specification of the west side.
    YOUNGER
        Indicates that a fault splits a genetic boundary on its younger
        side.
    OLDER
        Indicates that a fault splits a genetic boundary on its older side.
    BOTH
        Indicates that a fault splits both sides of a genetic feature.
    """

    FOOTWALL = "footwall"
    HANGING_WALL = "hanging wall"
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    YOUNGER = "younger"
    OLDER = "older"
    BOTH = "both"


class ContactVerb(Enum):
    """
    Enumerations for the verbs that can be used to define the impact on the
    construction of the model of the geological event that created the binary
    contact.

    Properties
    ----------
    SPLITS
        Specifies that the fault has opened a pair of fault lips in a
        horizon.
    INTERRUPTS
        Operation on which an “unconformable” genetic boundary
        interpretation interrupts another genetic boundary interpretation or
        a stratigraphic unit interpretation.
    CONTAINS
        Precise use of this attribute to be determined during testing.
    CONFORMS
        Defines surface contact between two stratigraphic units.
    ERODES
        Defines surface contact between two stratigraphic units.
    STOPS_AT
        Defines if a tectonic boundary interpretation stops at another
        tectonic boundary interpretation. Also used for genetic unit to
        frontier feature, fault to frontier feature, and sedimentary unit to
        frontier feature.
    CROSSES
        Defines if a tectonic boundary interpretation crosses another
        tectonic boundary interpretation.
    INCLUDES
        Precise use of this attribute will be determined during testing.
    """

    SPLITS = "splits"
    INTERRUPTS = "interrupts"
    CONTAINS = "contains"
    CONFORMS = "conforms"
    ERODES = "erodes"
    STOPS_AT = "stops at"
    CROSSES = "crosses"
    INCLUDES = "includes"


class DepositionMode(Enum):
    """
    The enumerated attributes of a horizon.
    """

    PROPORTIONAL_BETWEEN_TOP_AND_BOTTOM = "proportional between top and bottom"
    PARALLEL_TO_BOTTOM = "parallel to bottom"
    PARALLEL_TO_TOP = "parallel to top"
    PARALLEL_TO_ANOTHER_BOUNDARY = "parallel to another boundary"


class Domain(Enum):
    """
    Enumeration for the feature interpretation to specify if the measurement is in
    the seismic time or depth domain or if it is derived from a laboratory
    measurement.

    Properties
    ----------
    DEPTH
        Position defined by measurements in the depth domain.
    TIME
        Position based on geophysical measurements in two-way time (TWT).
    MIXED
    """

    DEPTH = "depth"
    TIME = "time"
    MIXED = "mixed"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DoubleLookup:
    """
    (key,value) pairs for a lookup table.

    Parameters
    ----------
    key
        Input to a table lookup.
    value
        Output from a table lookup.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    key: float = field(
        metadata={
            "name": "Key",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    value: float = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


class Facet(Enum):
    """Enumerations of the type of qualifier that applies to a property type to
    provide additional context about the nature of the property.

    For example, may include conditions, direction, qualifiers, or
    statistics. Facets are used in RESQML to provide qualifiers to
    existing property types, which minimizes the need to create
    specialized property types.

    Properties
    ----------
    CONDITIONS
        Indicates condition of how the property was acquired, e.g.,
        distinguishing surface condition of a fluid compared to reservoir
        conditions.
    DIRECTION
        Indicates that the property is directional. Common values are X, Y,
        or Z for vectors; I, J, or K for properties on a grid; or tensorial
        coordinates, e.g., XX or IJ. For example, vertical permeability vs.
        horizontal permeability.
    NETGROSS
        Indicates that the property is of kind net or gross, i.e., indicates
        that the spatial support of a property is averaged only over the net
        rock or all of the rock. rock or all of the rock.
    QUALIFIER
        Used to capture any other context not covered by the other facet
        types listed here.
    STATISTICS
        Indicates values such as minimum, maximum, average, etc.
    WHAT
        Indicates the element that is measured, for example, the
        concentration of a mineral.
    """

    CONDITIONS = "conditions"
    DIRECTION = "direction"
    NETGROSS = "netgross"
    QUALIFIER = "qualifier"
    STATISTICS = "statistics"
    WHAT = "what"


class FluidContact(Enum):
    """Enumerated values used to indicate a specific type of fluid boundary
    feature.

    See attributes below.

    Properties
    ----------
    FREE_WATER_CONTACT
        A surface defined by vanishing capillary pressure between the water
        and hydrocarbon phases.
    GAS_OIL_CONTACT
        A surface defined by vanishing capillary pressure between the gas
        and oil hydrocarbon phases.
    GAS_WATER_CONTACT
        A surface defined by vanishing capillary pressure between the water
        and gas hydrocarbon phases.
    SEAL
        Identifies a break in the hydrostatic column.
    WATER_OIL_CONTACT
        A surface defined by vanishing capillary pressure between the water
        and oil hydrocarbon phases.
    """

    FREE_WATER_CONTACT = "free water contact"
    GAS_OIL_CONTACT = "gas oil contact"
    GAS_WATER_CONTACT = "gas water contact"
    SEAL = "seal"
    WATER_OIL_CONTACT = "water oil contact"


class FluidMarker(Enum):
    """
    The various fluid a well marker can indicate.
    """

    GAS_DOWN_TO = "gas down to"
    GAS_UP_TO = "gas up to"
    OIL_DOWN_TO = "oil down to"
    OIL_UP_TO = "oil up to"
    WATER_DOWN_TO = "water down to"
    WATER_UP_TO = "water up to"


class GeneticBoundaryKind(Enum):
    """Enumerations used to indicate a specific type of genetic boundary feature.

    See attributes below. Note that a genetic boundary has a younger
    side and an older side.

    Properties
    ----------
    GEOBODY_BOUNDARY
        An interface between a geobody and other geologic objects.
    HORIZON
        An interface associated with a stratigraphic unit, which could be
        the top or bottom of the unit.
    """

    GEOBODY_BOUNDARY = "geobody boundary"
    HORIZON = "horizon"


class Geobody3dShape(Enum):
    """
    The enumerated attributes of a horizon.
    """

    DYKE = "dyke"
    SILT = "silt"
    DOME = "dome"
    SHEETH = "sheeth"
    DIAPIR = "diapir"
    BATHOLITH = "batholith"
    CHANNEL = "channel"
    DELTA = "delta"
    DUNE = "dune"
    FAN = "fan"
    REEF = "reef"
    WEDGE = "wedge"


class GeologicBoundaryKind(Enum):
    """
    The various geologic boundary a well marker can indicate.
    """

    FAULT = "fault"
    GEOBODY = "geobody"
    HORIZON = "horizon"


class GeologicUnitComposition(Enum):
    INTRUSIVE_CLAY = "intrusive clay "
    ORGANIC = "organic"
    INTRUSIVE_MUD = "intrusive mud "
    EVAPORITE_SALT = "evaporite salt"
    EVAPORITE_NON_SALT = "evaporite non salt"
    SEDIMENTARY_SILICLASTIC = "sedimentary siliclastic"
    CARBONATE = "carbonate"
    MAGMATIC_INTRUSIVE_GRANITOID = "magmatic intrusive granitoid"
    MAGMATIC_INTRUSIVE_PYROCLASTIC = "magmatic intrusive pyroclastic"
    MAGMATIC_EXTRUSIVE_LAVA_FLOW = "magmatic extrusive lava flow"
    OTHER_CHEMICHAL_ROCK = "other chemichal rock"
    SEDIMENTARY_TURBIDITE = "sedimentary turbidite"


class GeologicUnitMaterialImplacement(Enum):
    """
    The enumerated attributes of a horizon.
    """

    AUTOCHTONOUS = "autochtonous"
    ALLOCHTONOUS = "allochtonous"


class GridGeometryAttachment(Enum):
    """
    Indexable grid elements to which point geometry may be attached to describe
    additional grid geometry.

    Properties
    ----------
    CELLS
        Geometry may be attached to cells to distort the geometry of that
        specific cell, only (finite element grid).
    EDGES
        Geometry may be attached to edges to distort the geometry of all
        cells that refer to that edge (finite element grid). BUSINESS RULE:
        The edges indexing must be known or defined in the grid
        representation if geometry is attached to the edges.
    FACES
        Geometry may be attached to faces to distort the geometry of all
        cells that refer to that face (finite element grid). BUSINESS RULE:
        The faces indexing must be known or defined in the grid
        representation if geometry is attached to the faces.
    HINGE_NODE_FACES
        For column layer grids, these are the K faces. For unstructured
        grids these faces are enumerated as the hinge node faces.
    NODES
        Additional grid geometry may be attached to split or truncated node
        patches for column layer grids. All other node geometry attachment
        should be done through the Points array of the AbstractGridGeometry,
        not through the additional grid geometry.
    RADIAL_ORIGIN_POLYLINE
        NKL points must be attached to the radial origin polyline for a grid
        with radial interpolation. BUSINESS RULE: The optional
        radialGridIsComplete element must be defined in the grid
        representation if geometry is attached to the radial origin
        polyline.
    SUBNODES
        Geometry may be attached to subnodes to distort the geometry of all
        cells that refer to that subnode (finite element grid). BUSINESS
        RULE: An optional subnode patch object must be defined in the grid
        representation if geometry is attached to the subnodes.
    """

    CELLS = "cells"
    EDGES = "edges"
    FACES = "faces"
    HINGE_NODE_FACES = "hinge node faces"
    NODES = "nodes"
    RADIAL_ORIGIN_POLYLINE = "radial origin polyline"
    SUBNODES = "subnodes"


class IdentityKind(Enum):
    """
    Enumeration of the identity kinds for the element identities (ElementIdentity).

    Properties
    ----------
    COLOCATION
        A set of (sub)representations is collocated if there is bijection
        between the simple elements of all of the participating
        (sub)representations. This definition implies there is the same
        number of simple elements. NOTE: The geometric location of each set
        of simple elements mapped through the bijection is intended to be
        identical even if the numeric values of the associated geometries
        differ, i.e., due to loss of spatial resolution.
    PREVIOUS_COLOCATION
        The participating (sub)representations were collocated at some time
        in the geologic past—but not necessarily in the present day earth
        model.
    EQUIVALENCE
        A set of (sub)representations is equivalent if there is a map giving
        an association between some of the simple topological elements of
        the participating (sub)representations.
    PREVIOUS_EQUIVALENCE
        The participating (sub)representations were equivalent at some time
        in the geologic past—but not necessarily in the present day earth
        model.
    """

    COLOCATION = "colocation"
    PREVIOUS_COLOCATION = "previous colocation"
    EQUIVALENCE = "equivalence"
    PREVIOUS_EQUIVALENCE = "previous equivalence"


class IndexableElements(Enum):
    """Indexable elements for the different representations. The indexing of each
    element depends upon the specific representation. To order and reference the
    elements of a representation, RESQML makes extensive use of the concept of
    indexing. Both one-dimensional and multi-dimensional arrays of elements are
    used. So that all elements may be referenced in a consistent and uniform
    fashion, each multi-dimensional index must have a well-defined 1D index.
    Attributes below identify the IndexableElements, though not all elements apply
    to all types of representations.

    Indexable elements are used to:
    - attach geometry and properties to a representation.
    - identify portions of a representation when expressing a representation identity.
    - construct a sub-representation from an existing representation.
    See the RESQML Overview Guide for the table of indexable elements and the representations to which they apply.

    Properties
    ----------
    CELLS
    COLUMN_EDGES
    COLUMNS
    CONTACTS
    COORDINATE_LINES
    EDGES
    EDGES_PER_COLUMN
    ENUMERATED_ELEMENTS
    FACES
    FACES_PER_CELL
    INTERVAL_EDGES
        Count = NKL (Column Layer Grids, only)
    INTERVALS
    I0
        Count = NI (IJK Grids, only)
    I0_EDGES
        Count = NIL (IJK Grids, only)
    J0
        Count = NJ (IJK Grids, only)
    J0_EDGES
        Count = NJL (IJK Grids, only)
    LAYERS
        Count = NK  (Column Layer Grids, only)
    NODES
    NODES_PER_CELL
    NODES_PER_EDGE
    NODES_PER_FACE
    PATCHES
    PILLARS
    REGIONS
    REPRESENTATION
    SUBNODES
    TRIANGLES
    """

    CELLS = "cells"
    COLUMN_EDGES = "column edges"
    COLUMNS = "columns"
    CONTACTS = "contacts"
    COORDINATE_LINES = "coordinate lines"
    EDGES = "edges"
    EDGES_PER_COLUMN = "edges per column"
    ENUMERATED_ELEMENTS = "enumerated elements"
    FACES = "faces"
    FACES_PER_CELL = "faces per cell"
    INTERVAL_EDGES = "interval edges"
    INTERVALS = "intervals"
    I0 = "I0"
    I0_EDGES = "I0 edges"
    J0 = "J0"
    J0_EDGES = "J0 edges"
    LAYERS = "layers"
    NODES = "nodes"
    NODES_PER_CELL = "nodes per cell"
    NODES_PER_EDGE = "nodes per edge"
    NODES_PER_FACE = "nodes per face"
    PATCHES = "patches"
    PILLARS = "pillars"
    REGIONS = "regions"
    REPRESENTATION = "representation"
    SUBNODES = "subnodes"
    TRIANGLES = "triangles"


class KDirection(Enum):
    """Enumeration used to specify if the direction of the coordinate lines is
    uniquely defined for a grid.

    If not uniquely defined, e.g., for over-turned reservoirs, then
    indicate that the K direction is not monotonic.

    Properties
    ----------
    DOWN
        K is increasing with depth, dot(tangent,gradDepth)&gt;0.
    UP
        K is increasing with elevation, dot(tangent,gradDepth)&lt;0.
    NOT_MONOTONIC
        K is not monotonic with elevation, e.g., for over-turned structures.
    """

    DOWN = "down"
    UP = "up"
    NOT_MONOTONIC = "not monotonic"


class LineRole(Enum):
    """
    Indicates the various roles that a polyline topology can have in a
    representation.

    Properties
    ----------
    FAULT_CENTER_LINE
        Usually used to represent fault lineaments on horizons. These lines
        can represent nonsealed contact interpretation parts defined by a
        horizon/fault intersection.
    PICK
        Used to represent all types of nonsealed contact interpretation
        parts defined by a horizon/fault intersection.
    INNER_RING
        Closed polyline that delineates a hole in a surface patch.
    OUTER_RING
        Closed polyline that delineates the extension of a surface patch.
    TRAJECTORY
        Polyline that is used to represent a well trajectory representation.
    INTERPRETATION_LINE
        Line corresponding to a digitalization along an earth model section.
    CONTACT
        Used to represent nonsealed contact interpretation parts defined by
        a horizon/fault intersection.
    DEPOSITIONAL_LINE
        Used to represent nonsealed contact interpretation parts defined by
        a horizon/horizon intersection.
    EROSION_LINE
        Used to represent nonsealed contact interpretation parts defined by
        a horizon/horizon intersection.
    CONTOURING
        Used to obtain sets of 3D x, y, z points to represent any boundary
        interpretation.
    PILLAR
    """

    FAULT_CENTER_LINE = "fault center line"
    PICK = "pick"
    INNER_RING = "inner ring"
    OUTER_RING = "outer ring"
    TRAJECTORY = "trajectory"
    INTERPRETATION_LINE = "interpretation line"
    CONTACT = "contact"
    DEPOSITIONAL_LINE = "depositional line"
    EROSION_LINE = "erosion line"
    CONTOURING = "contouring"
    PILLAR = "pillar"


class MdDomain(Enum):
    """
    Different types of measured depths.

    Properties
    ----------
    DRILLER
        The original depths recorded while drilling a well or LWD or MWD.
    LOGGER
        Depths recorded when logging a well, which are in general considered
        to be more accurate than driller's depth.
    """

    DRILLER = "driller"
    LOGGER = "logger"


class MdReference(Enum):
    """Reference location for the measured depth datum (MdDatum).

    The type of local or permanent reference datum for vertical gravity
    based (i.e., elevation and vertical depth) and measured depth
    coordinates within the context of a well. This list includes local
    points (e.g., kelly bushing) used as a datum and vertical reference
    datums (e.g., mean sea level).

    Properties
    ----------
    GROUND_LEVEL
    KELLY_BUSHING
    MEAN_SEA_LEVEL
        A tidal datum. The arithmetic mean of hourly heights observed over
        the National Tidal Datum Epoch (19 years).
    DERRICK_FLOOR
    CASING_FLANGE
        A flange affixed to the top of the casing string used to attach
        production equipment.
    ARBITRARY_POINT
        This value should not be used for drilled wells. All reasonable
        attempts should be made to determine the appropriate value.
    CROWN_VALVE
    ROTARY_BUSHING
    ROTARY_TABLE
    SEA_FLOOR
    LOWEST_ASTRONOMICAL_TIDE
        The lowest tide level over the duration of the National Tidal Datum
        Epoch (19 years).
    MEAN_HIGHER_HIGH_WATER
        A tidal datum. The average of the higher high water height of each
        tidal day observed over the National Tidal Datum Epoch (19 years).
    MEAN_HIGH_WATER
        A tidal datum. The average of all the high water heights observed
        over the National Tidal Datum Epoch (19 years).
    MEAN_LOWER_LOW_WATER
        A tidal datum. The average of the lower low water height of each
        tidal day observed over the National Tidal Datum Epoch (19 years ).
    MEAN_LOW_WATER
        A tidal datum. The average of all the low water heights observed
        over the National Tidal Datum Epoch (19 years).
    MEAN_TIDE_LEVEL
        A tidal datum. The arithmetic mean of mean high water and mean low
        water. Same as half-tide level.
    KICKOFF_POINT
        This value is not expected to be used in most typical situations.
        All reasonable attempts should be made to determine the appropriate
        value.
    """

    GROUND_LEVEL = "ground level"
    KELLY_BUSHING = "kelly bushing"
    MEAN_SEA_LEVEL = "mean sea level"
    DERRICK_FLOOR = "derrick floor"
    CASING_FLANGE = "casing flange"
    ARBITRARY_POINT = "arbitrary point"
    CROWN_VALVE = "crown valve"
    ROTARY_BUSHING = "rotary bushing"
    ROTARY_TABLE = "rotary table"
    SEA_FLOOR = "sea floor"
    LOWEST_ASTRONOMICAL_TIDE = "lowest astronomical tide"
    MEAN_HIGHER_HIGH_WATER = "mean higher high water"
    MEAN_HIGH_WATER = "mean high water"
    MEAN_LOWER_LOW_WATER = "mean lower low water"
    MEAN_LOW_WATER = "mean low water"
    MEAN_TIDE_LEVEL = "mean tide level"
    KICKOFF_POINT = "kickoff point"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class NameValuePair:
    """
    Complementary optional metadata information, which may be added to any data-
    object to qualify it.

    Parameters
    ----------
    name
        Name of the metadata information.
    value
        Value of the metadata information.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    name: str = field(
        metadata={
            "name": "Name",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    value: str = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


class OrderingCriteria(Enum):
    """
    Enumeration used to specify the order of an abstract stratigraphic organization
    or a structural organization interpretation.

    Properties
    ----------
    AGE
        From youngest to oldest period (increasing age).
    APPARENT_DEPTH
        From surface to subsurface.
    MEASURED_DEPTH
        From well head to wellbore bottom/total depth (TD).
    """

    AGE = "age"
    APPARENT_DEPTH = "apparent depth"
    MEASURED_DEPTH = "measured depth"


class OrganizationKind(Enum):
    """Enumerations used to indicate a specific type of organization.

    See attributes below.

    Properties
    ----------
    EARTH_MODEL
        An organization composed of the other types of organizations listed
        here.
    FLUID
        A volume organization composed of fluid boundaries and phase units.
    STRATIGRAPHIC
        A volume organization composed of geologic features, such as
        geobodies, stratigraphic units, and boundaries.
    STRUCTURAL
        A surface organization composed of geologic features, such as
        faults, horizons, and frontier boundaries.
    """

    EARTH_MODEL = "earth model"
    FLUID = "fluid"
    STRATIGRAPHIC = "stratigraphic"
    STRUCTURAL = "structural"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class OrientedMacroFace:
    """An element of a volume shell that is defined by a set of oriented faces
    belonging to boundable patches.

    A macroface may describe a contact between:
    - two structural, stratigraphic, or fluid units
    - one boundary feature (fault or frontier) and a unit.
    A face is a bounded open subset of a plane or a curved surface in 3D, delimited by an outer contour and zero, one, or more inner contours describing holes.

    Parameters
    ----------
    patch_index_of_representation
        Create the triangulation and 2D grid representation for which the
        patches match the macro faces.
    representation_index
        Identifies the representation by its index, in the list of
        representations contained in the organization.
    side_is_plus
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_index_of_representation: int = field(
        metadata={
            "name": "PatchIndexOfRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation_index: int = field(
        metadata={
            "name": "RepresentationIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    side_is_plus: bool = field(
        metadata={
            "name": "SideIsPlus",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


class ParameterKind(Enum):
    DATA_OBJECT = "dataObject"
    FLOATING_POINT = "floatingPoint"
    INTEGER = "integer"
    STRING = "string"
    TIMESTAMP = "timestamp"
    SUB_ACTIVITY = "subActivity"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Patch:
    """Set or range of one kind of topological element used to define part of a
    data-object; this concept exists for grid and structural data-objects.

    Subset of a specified kind of indexable element of a representation,
    associated with a patch index. The patch index is used to define the
    relative order for the elements. It may also be used to access
    patches of indexable elements directly for geometry, properties, or
    relationships. Patches are used to remove any ambiguity in data
    ordering among the indexable elements. For example, the triangle
    indexing of a triangulated set representation consists of multiple
    triangles, each with a patch index. This index specifies the
    relative ordering of the triangle patches. Those data-objects that
    inherit a patch index from the abstract class of patches all include
    the word “Patch” as part of their name, e.g., TrianglePatch.

    Parameters
    ----------
    patch_index
        Local index of the patch, making it unique within the
        representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_index: int = field(
        metadata={
            "name": "PatchIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


class Phase(Enum):
    """The enumeration of the possible RockFluid Unit phase  in a hydrostatic
    column.

    The seal is considered here as a part ( the coverage phase) of an
    hydrostatic column.

    Properties
    ----------
    AQUIFER
        Volume of the hydrostatic column for which only the aqueous phase is
        mobile. Typically below the Pc(hydrocarbon-water)=0 free fluid
        surface.
    GAS_CAP
        Volume of the hydrostatic column for which only the gaseous phase is
        mobile. Typically above the Pc(gas-oil)=0 free fluid surface.
    OIL_COLUMN
        Volume of the hydrostatic column for which only the oleic and
        aqueous phases may be mobile. Typically below the gas-oil Pc=0 free
        fluid surface.Pc(gas-oil)=0 free fluid surface.
    SEAL
        Impermeable volume which provides the seal for a hydrostatic fluid
        column.
    """

    AQUIFER = "aquifer"
    GAS_CAP = "gas cap"
    OIL_COLUMN = "oil column"
    SEAL = "seal"


class PillarShape(Enum):
    """Used to indicate that all pillars are of a uniform kind, i.e., may be
    represented using the same number of nodes per pillar.

    This information is supplied by the RESQML writer to indicate the complexity of the grid geometry, as an aide to the RESQML reader.
    If a combination of vertical and straight, then use straight.
    If a specific pillar shape is not appropriate, then use curved.
    BUSINESS RULE: Should be consistent with the actual geometry of the grid.

    Properties
    ----------
    VERTICAL
        If represented by a parametric line, requires only a single control
        point per line.
    STRAIGHT
        If represented by a parametric line, requires 2 control points per
        line.
    CURVED
        If represented by a parametric line, requires 3 or more control
        points per line.
    """

    VERTICAL = "vertical"
    STRAIGHT = "straight"
    CURVED = "curved"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3d:
    """
    Defines a point using coordinates in 3D space.

    Parameters
    ----------
    coordinate1
        X Coordinate
    coordinate2
        Y Coordinate
    coordinate3
        Either Z or T Coordinate
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    coordinate1: float = field(
        metadata={
            "name": "Coordinate1",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    coordinate2: float = field(
        metadata={
            "name": "Coordinate2",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    coordinate3: float = field(
        metadata={
            "name": "Coordinate3",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


class ResqmlPropertyKind(Enum):
    """
    Enumeration of the standard set of RESQML property kinds.

    Properties
    ----------
    ABSORBED_DOSE
        The amount of energy absorbed per mass.
    ACCELERATION_LINEAR
    ACTIVITY_OF_RADIOACTIVITY
        A measure of the radiation being emitted.
    AMOUNT_OF_SUBSTANCE
        Molar amount of a substance.
    AMPLITUDE
        Amplitude of the acoustic signal recorded. It is not a physical
        property, only a value.
    ANGLE_PER_LENGTH
    ANGLE_PER_TIME
        The angular velocity. The rate of change of an angle.
    ANGLE_PER_VOLUME
    ANGULAR_ACCELERATION
    AREA
    AREA_PER_AREA
        A dimensionless quantity where the basis of the ratio is area.
    AREA_PER_VOLUME
    ATTENUATION
        A logarithmic, fractional change of some measure, generally power or
        amplitude, over a standard range. This is generally used for
        frequency attenuation over an octave.
    ATTENUATION_PER_LENGTH
    AZIMUTH
        Angle between the North and the projection of the normal to the
        horizon surface estimated on a local area of the interface.
    BUBBLE_POINT_PRESSURE
        The pressure at which the first gas bubble appears while decreasing
        pressure on a fluid sample.
    BULK_MODULUS
        Bulk modulus, K
    CAPACITANCE
    CATEGORICAL
        The abstract supertype of all enumerated string properties.
    CELL_LENGTH
        distance from cell face center to cell face center in the specified
        direction, DI, DJ, DK
    CHARGE_DENSITY
    CHEMICAL_POTENTIAL
    CODE
        A discrete code.
    COMPRESSIBILITY
    CONCENTRATION_OF_B
        molar concentration of a substance.
    CONDUCTIVITY
    CONTINUOUS
        The abstract supertype of all floating point properties.
    CROSS_SECTION_ABSORPTION
    CURRENT_DENSITY
    DARCY_FLOW_COEFFICIENT
    DATA_TRANSMISSION_SPEED
        used primarily for computer transmission rates.
    DELTA_TEMPERATURE
        Delta temperature refers to temperature differences. For non-zero
        offset temperature scales, Fahrenheit and Celsius, the conversion
        formulas are different than for absolute temperatures.
    DENSITY
    DEPTH
        The perpendicular measurement downward from a surface. Also, the
        direct linear measurement from the point of viewing usually from
        front to back.
    DIFFUSION_COEFFICIENT
    DIGITAL_STORAGE
    DIMENSIONLESS
        A dimensionless quantity is the ratio of two dimensional quantities.
        The quantity types are not apparent from the basic dimensionless
        class, but may be apparent in variations - such as area per area,
        volume per volume, or mass per mass.
    DIP
        In the azimuth direction, Angle between an horizon plane and an
        estimated plane on a local area of the interface.
    DISCRETE
        The abstract supertype of all integer properties.
    DOSE_EQUIVALENT
    DOSE_EQUIVALENT_RATE
    DYNAMIC_VISCOSITY
    ELECTRIC_CHARGE
    ELECTRIC_CONDUCTANCE
    ELECTRIC_CURRENT
    ELECTRIC_DIPOLE_MOMENT
    ELECTRIC_FIELD_STRENGTH
    ELECTRIC_POLARIZATION
    ELECTRIC_POTENTIAL
    ELECTRICAL_RESISTIVITY
    ELECTROCHEMICAL_EQUIVALENT
        An electrochemical equivalent differs from molarity in that the
        valence (oxidation reduction potential) of the element is also
        considered.
    ELECTROMAGNETIC_MOMENT
    ENERGY_LENGTH_PER_AREA
    ENERGY_LENGTH_PER_TIME_AREA_TEMPERATURE
    ENERGY_PER_AREA
    ENERGY_PER_LENGTH
    EQUIVALENT_PER_MASS
    EQUIVALENT_PER_VOLUME
    EXPOSURE_RADIOACTIVITY
    FLUID_VOLUME
        Volume of fluid
    FORCE
    FORCE_AREA
    FORCE_LENGTH_PER_LENGTH
    FORCE_PER_FORCE
        A dimensionless quantity where the basis of the ratio is force.
    FORCE_PER_LENGTH
    FORCE_PER_VOLUME
    FORMATION_VOLUME_FACTOR
        Ratio of volumes at subsurface and surface conditions
    FREQUENCY
    FREQUENCY_INTERVAL
        An octave is a doubling of a frquency.
    GAMMA_RAY_API_UNIT
        This class is defined by the API, and is used for units of gamma ray
        log response.
    HEAT_CAPACITY
    HEAT_FLOW_RATE
    HEAT_TRANSFER_COEFFICIENT
        PRESSURE PER VELOCITY PER AREA
    ILLUMINANCE
    INDEX
        Serial ordering
    IRRADIANCE
    ISOTHERMAL_COMPRESSIBILITY
    KINEMATIC_VISCOSITY
    LAMBDA_RHO
        Product of Lame constant and density, LR
    LAME_CONSTANT
        Lame constant, Lambda
    LENGTH
    LENGTH_PER_LENGTH
        A dimensionless quantity where the basis of the ratio is length.
    LENGTH_PER_TEMPERATURE
    LENGTH_PER_VOLUME
    LEVEL_OF_POWER_INTENSITY
    LIGHT_EXPOSURE
    LINEAR_THERMAL_EXPANSION
    LUMINANCE
    LUMINOUS_EFFICACY
    LUMINOUS_FLUX
    LUMINOUS_INTENSITY
    MAGNETIC_DIPOLE_MOMENT
    MAGNETIC_FIELD_STRENGTH
    MAGNETIC_FLUX
    MAGNETIC_INDUCTION
    MAGNETIC_PERMEABILITY
    MAGNETIC_VECTOR_POTENTIAL
    MASS
        M/L2T
    MASS_ATTENUATION_COEFFICIENT
    MASS_CONCENTRATION
        A dimensionless quantity where the basis of the ratio is mass.
    MASS_FLOW_RATE
    MASS_LENGTH
    MASS_PER_ENERGY
    MASS_PER_LENGTH
        M /L4T
    MASS_PER_TIME_PER_AREA
    MASS_PER_TIME_PER_LENGTH
    MASS_PER_VOLUME_PER_LENGTH
    MOBILITY
    MODULUS_OF_COMPRESSION
    MOLAR_CONCENTRATION
        molar concentration of a substance.
    MOLAR_HEAT_CAPACITY
    MOLAR_VOLUME
    MOLE_PER_AREA
    MOLE_PER_TIME
    MOLE_PER_TIME_PER_AREA
    MOMENT_OF_FORCE
    MOMENT_OF_INERTIA
    MOMENT_OF_SECTION
    MOMENTUM
    MU_RHO
        Product of Shear modulus and density, MR
    NET_TO_GROSS_RATIO
        Ratio of net rock volume to gross rock volume, NTG
    NEUTRON_API_UNIT
    NON_DARCY_FLOW_COEFFICIENT
    OPERATIONS_PER_TIME
    PARACHOR
    PER_AREA
    PER_ELECTRIC_POTENTIAL
    PER_FORCE
    PER_LENGTH
    PER_MASS
    PER_VOLUME
    PERMEABILITY_LENGTH
    PERMEABILITY_ROCK
    PERMEABILITY_THICKNESS
        Product of permeability and thickness
    PERMEANCE
    PERMITTIVITY
    P_H
        The pH is a class that measures the hydrogen ion concentration
        (acidity).
    PLANE_ANGLE
    POISSON_RATIO
        Poisson's ratio, Sigma
    PORE_VOLUME
        Volume of the Pore Space of the Rock
    POROSITY
        porosity
    POTENTIAL_DIFFERENCE_PER_POWER_DROP
    POWER
    POWER_PER_VOLUME
    PRESSURE
    PRESSURE_PER_TIME
    PRESSURE_SQUARED
    PRESSURE_SQUARED_PER_FORCE_TIME_PER_AREA
    PRESSURE_TIME_PER_VOLUME
    PRODUCTIVITY_INDEX
    PROPERTY_MULTIPLIER
        Unitless multiplier to apply to any property
    QUANTITY
        The abstract supertype of all floating point properties with a unit
        of measure.
    QUANTITY_OF_LIGHT
    RADIANCE
    RADIANT_INTENSITY
    RELATIVE_PERMEABILITY
        Ratio of phase permeability, which is a function of saturation, to
        the rock permeability
    RELATIVE_POWER
        A dimensionless quantity where the basis of the ratio is power.
    RELATIVE_TIME
        A dimensionless quantity where the basis of the ratio is time.
    RELUCTANCE
    RESISTANCE
    RESISTIVITY_PER_LENGTH
    RESQML_ROOT_PROPERTY
        The abstract supertype of all properties. This property does not
        have a parent.
    ROCK_IMPEDANCE
        Acoustic impedence, Ip, Is
    ROCK_PERMEABILITY
        See "permeability rock"
    ROCK_VOLUME
        Rock Volume
    SATURATION
        Ratio of phase fluid volume to pore volume
    SECOND_MOMENT_OF_AREA
    SHEAR_MODULUS
        Shear modulus, Mu
    SOLID_ANGLE
    SOLUTION_GAS_OIL_RATIO
        Ratio of solution gas volume to oil volume at reservoir conditions
    SPECIFIC_ACTIVITY_OF_RADIOACTIVITY
    SPECIFIC_ENERGY
    SPECIFIC_HEAT_CAPACITY
    SPECIFIC_PRODUCTIVITY_INDEX
    SPECIFIC_VOLUME
    SURFACE_DENSITY
    TEMPERATURE_PER_LENGTH
    TEMPERATURE_PER_TIME
    THERMAL_CONDUCTANCE
    THERMAL_CONDUCTIVITY
    THERMAL_DIFFUSIVITY
    THERMAL_INSULANCE
    THERMAL_RESISTANCE
    THERMODYNAMIC_TEMPERATURE
    THICKNESS
        Distance measured in a volume between two surfaces ( e.G. Geological
        Top Boundary and Geological Bottom Boundary of a Geological unit).
    TIME
    TIME_PER_LENGTH
    TIME_PER_VOLUME
    TRANSMISSIBILITY
        Volumetric flux per unit area per unit pressure drop for unit
        viscosity fluid
    UNIT_PRODUCTIVITY_INDEX
    UNITLESS
        The abstract supertype of all floating point properties with NO unit
        of measure. In order to allow the unit information to be required
        for all continuous properties, the special unit of measure of "NONE"
        has been assigned to all children of this class. In addition, the
        special dimensional class of "0" has been assigned to all children
        of this class.
    VAPOR_OIL_GAS_RATIO
        Ratio of oil vapor volume to gas volume at reservoir conditions
    VELOCITY
    VOLUME
    VOLUME_FLOW_RATE
    VOLUME_LENGTH_PER_TIME
    VOLUME_PER_AREA
    VOLUME_PER_LENGTH
    VOLUME_PER_TIME_PER_AREA
    VOLUME_PER_TIME_PER_LENGTH
    VOLUME_PER_TIME_PER_TIME
    VOLUME_PER_TIME_PER_VOLUME
    VOLUME_PER_VOLUME
        A dimensionless quantity where the basis of the ratio is volume.
    VOLUMETRIC_HEAT_TRANSFER_COEFFICIENT
    VOLUMETRIC_THERMAL_EXPANSION
    WORK
    YOUNG_MODULUS
        Young's modulus, E
    """

    ABSORBED_DOSE = "absorbed dose"
    ACCELERATION_LINEAR = "acceleration linear"
    ACTIVITY_OF_RADIOACTIVITY = "activity (of radioactivity)"
    AMOUNT_OF_SUBSTANCE = "amount of substance"
    AMPLITUDE = "amplitude"
    ANGLE_PER_LENGTH = "angle per length"
    ANGLE_PER_TIME = "angle per time"
    ANGLE_PER_VOLUME = "angle per volume"
    ANGULAR_ACCELERATION = "angular acceleration"
    AREA = "area"
    AREA_PER_AREA = "area per area"
    AREA_PER_VOLUME = "area per volume"
    ATTENUATION = "attenuation"
    ATTENUATION_PER_LENGTH = "attenuation per length"
    AZIMUTH = "azimuth"
    BUBBLE_POINT_PRESSURE = "bubble point pressure"
    BULK_MODULUS = "bulk modulus"
    CAPACITANCE = "capacitance"
    CATEGORICAL = "categorical"
    CELL_LENGTH = "cell length"
    CHARGE_DENSITY = "charge density"
    CHEMICAL_POTENTIAL = "chemical potential"
    CODE = "code"
    COMPRESSIBILITY = "compressibility"
    CONCENTRATION_OF_B = "concentration of B"
    CONDUCTIVITY = "conductivity"
    CONTINUOUS = "continuous"
    CROSS_SECTION_ABSORPTION = "cross section absorption"
    CURRENT_DENSITY = "current density"
    DARCY_FLOW_COEFFICIENT = "Darcy flow coefficient"
    DATA_TRANSMISSION_SPEED = "data transmission speed"
    DELTA_TEMPERATURE = "delta temperature"
    DENSITY = "density"
    DEPTH = "depth"
    DIFFUSION_COEFFICIENT = "diffusion coefficient"
    DIGITAL_STORAGE = "digital storage"
    DIMENSIONLESS = "dimensionless"
    DIP = "dip"
    DISCRETE = "discrete"
    DOSE_EQUIVALENT = "dose equivalent"
    DOSE_EQUIVALENT_RATE = "dose equivalent rate"
    DYNAMIC_VISCOSITY = "dynamic viscosity"
    ELECTRIC_CHARGE = "electric charge"
    ELECTRIC_CONDUCTANCE = "electric conductance"
    ELECTRIC_CURRENT = "electric current"
    ELECTRIC_DIPOLE_MOMENT = "electric dipole moment"
    ELECTRIC_FIELD_STRENGTH = "electric field strength"
    ELECTRIC_POLARIZATION = "electric polarization"
    ELECTRIC_POTENTIAL = "electric potential"
    ELECTRICAL_RESISTIVITY = "electrical resistivity"
    ELECTROCHEMICAL_EQUIVALENT = "electrochemical equivalent"
    ELECTROMAGNETIC_MOMENT = "electromagnetic moment"
    ENERGY_LENGTH_PER_AREA = "energy length per area"
    ENERGY_LENGTH_PER_TIME_AREA_TEMPERATURE = "energy length per time area temperature"
    ENERGY_PER_AREA = "energy per area"
    ENERGY_PER_LENGTH = "energy per length"
    EQUIVALENT_PER_MASS = "equivalent per mass"
    EQUIVALENT_PER_VOLUME = "equivalent per volume"
    EXPOSURE_RADIOACTIVITY = "exposure (radioactivity)"
    FLUID_VOLUME = "fluid volume"
    FORCE = "force"
    FORCE_AREA = "force area"
    FORCE_LENGTH_PER_LENGTH = "force length per length"
    FORCE_PER_FORCE = "force per force"
    FORCE_PER_LENGTH = "force per length"
    FORCE_PER_VOLUME = "force per volume"
    FORMATION_VOLUME_FACTOR = "formation volume factor"
    FREQUENCY = "frequency"
    FREQUENCY_INTERVAL = "frequency interval"
    GAMMA_RAY_API_UNIT = "gamma ray API unit"
    HEAT_CAPACITY = "heat capacity"
    HEAT_FLOW_RATE = "heat flow rate"
    HEAT_TRANSFER_COEFFICIENT = "heat transfer coefficient"
    ILLUMINANCE = "illuminance"
    INDEX = "index"
    IRRADIANCE = "irradiance"
    ISOTHERMAL_COMPRESSIBILITY = "isothermal compressibility"
    KINEMATIC_VISCOSITY = "kinematic viscosity"
    LAMBDA_RHO = "Lambda Rho"
    LAME_CONSTANT = "Lame constant"
    LENGTH = "length"
    LENGTH_PER_LENGTH = "length per length"
    LENGTH_PER_TEMPERATURE = "length per temperature"
    LENGTH_PER_VOLUME = "length per volume"
    LEVEL_OF_POWER_INTENSITY = "level of power intensity"
    LIGHT_EXPOSURE = "light exposure"
    LINEAR_THERMAL_EXPANSION = "linear thermal expansion"
    LUMINANCE = "luminance"
    LUMINOUS_EFFICACY = "luminous efficacy"
    LUMINOUS_FLUX = "luminous flux"
    LUMINOUS_INTENSITY = "luminous intensity"
    MAGNETIC_DIPOLE_MOMENT = "magnetic dipole moment"
    MAGNETIC_FIELD_STRENGTH = "magnetic field strength"
    MAGNETIC_FLUX = "magnetic flux"
    MAGNETIC_INDUCTION = "magnetic induction"
    MAGNETIC_PERMEABILITY = "magnetic permeability"
    MAGNETIC_VECTOR_POTENTIAL = "magnetic vector potential"
    MASS = "mass"
    MASS_ATTENUATION_COEFFICIENT = "mass attenuation coefficient"
    MASS_CONCENTRATION = "mass concentration"
    MASS_FLOW_RATE = "mass flow rate"
    MASS_LENGTH = "mass length"
    MASS_PER_ENERGY = "mass per energy"
    MASS_PER_LENGTH = "mass per length"
    MASS_PER_TIME_PER_AREA = "mass per time per area"
    MASS_PER_TIME_PER_LENGTH = "mass per time per length"
    MASS_PER_VOLUME_PER_LENGTH = "mass per volume per length"
    MOBILITY = "mobility"
    MODULUS_OF_COMPRESSION = "modulus of compression"
    MOLAR_CONCENTRATION = "molar concentration"
    MOLAR_HEAT_CAPACITY = "molar heat capacity"
    MOLAR_VOLUME = "molar volume"
    MOLE_PER_AREA = "mole per area"
    MOLE_PER_TIME = "mole per time"
    MOLE_PER_TIME_PER_AREA = "mole per time per area"
    MOMENT_OF_FORCE = "moment of force"
    MOMENT_OF_INERTIA = "moment of inertia"
    MOMENT_OF_SECTION = "moment of section"
    MOMENTUM = "momentum"
    MU_RHO = "Mu Rho"
    NET_TO_GROSS_RATIO = "net to gross ratio"
    NEUTRON_API_UNIT = "neutron API unit"
    NON_DARCY_FLOW_COEFFICIENT = "nonDarcy flow coefficient"
    OPERATIONS_PER_TIME = "operations per time"
    PARACHOR = "parachor"
    PER_AREA = "per area"
    PER_ELECTRIC_POTENTIAL = "per electric potential"
    PER_FORCE = "per force"
    PER_LENGTH = "per length"
    PER_MASS = "per mass"
    PER_VOLUME = "per volume"
    PERMEABILITY_LENGTH = "permeability length"
    PERMEABILITY_ROCK = "permeability rock"
    PERMEABILITY_THICKNESS = "permeability thickness"
    PERMEANCE = "permeance"
    PERMITTIVITY = "permittivity"
    P_H = "pH"
    PLANE_ANGLE = "plane angle"
    POISSON_RATIO = "Poisson ratio"
    PORE_VOLUME = "pore volume"
    POROSITY = "porosity"
    POTENTIAL_DIFFERENCE_PER_POWER_DROP = "potential difference per power drop"
    POWER = "power"
    POWER_PER_VOLUME = "power per volume"
    PRESSURE = "pressure"
    PRESSURE_PER_TIME = "pressure per time"
    PRESSURE_SQUARED = "pressure squared"
    PRESSURE_SQUARED_PER_FORCE_TIME_PER_AREA = (
        "pressure squared per force time per area"
    )
    PRESSURE_TIME_PER_VOLUME = "pressure time per volume"
    PRODUCTIVITY_INDEX = "productivity index"
    PROPERTY_MULTIPLIER = "property multiplier"
    QUANTITY = "quantity"
    QUANTITY_OF_LIGHT = "quantity of light"
    RADIANCE = "radiance"
    RADIANT_INTENSITY = "radiant intensity"
    RELATIVE_PERMEABILITY = "relative permeability"
    RELATIVE_POWER = "relative power"
    RELATIVE_TIME = "relative time"
    RELUCTANCE = "reluctance"
    RESISTANCE = "resistance"
    RESISTIVITY_PER_LENGTH = "resistivity per length"
    RESQML_ROOT_PROPERTY = "RESQML root property"
    ROCK_IMPEDANCE = "Rock Impedance"
    ROCK_PERMEABILITY = "rock permeability"
    ROCK_VOLUME = "rock volume"
    SATURATION = "saturation"
    SECOND_MOMENT_OF_AREA = "second moment of area"
    SHEAR_MODULUS = "shear modulus"
    SOLID_ANGLE = "solid angle"
    SOLUTION_GAS_OIL_RATIO = "solution gas-oil ratio"
    SPECIFIC_ACTIVITY_OF_RADIOACTIVITY = "specific activity (of radioactivity)"
    SPECIFIC_ENERGY = "specific energy"
    SPECIFIC_HEAT_CAPACITY = "specific heat capacity"
    SPECIFIC_PRODUCTIVITY_INDEX = "specific productivity index"
    SPECIFIC_VOLUME = "specific volume"
    SURFACE_DENSITY = "surface density"
    TEMPERATURE_PER_LENGTH = "temperature per length"
    TEMPERATURE_PER_TIME = "temperature per time"
    THERMAL_CONDUCTANCE = "thermal conductance"
    THERMAL_CONDUCTIVITY = "thermal conductivity"
    THERMAL_DIFFUSIVITY = "thermal diffusivity"
    THERMAL_INSULANCE = "thermal insulance"
    THERMAL_RESISTANCE = "thermal resistance"
    THERMODYNAMIC_TEMPERATURE = "thermodynamic temperature"
    THICKNESS = "thickness"
    TIME = "time"
    TIME_PER_LENGTH = "time per length"
    TIME_PER_VOLUME = "time per volume"
    TRANSMISSIBILITY = "transmissibility"
    UNIT_PRODUCTIVITY_INDEX = "unit productivity index"
    UNITLESS = "unitless"
    VAPOR_OIL_GAS_RATIO = "vapor oil-gas ratio"
    VELOCITY = "velocity"
    VOLUME = "volume"
    VOLUME_FLOW_RATE = "volume flow rate"
    VOLUME_LENGTH_PER_TIME = "volume length per time"
    VOLUME_PER_AREA = "volume per area"
    VOLUME_PER_LENGTH = "volume per length"
    VOLUME_PER_TIME_PER_AREA = "volume per time per area"
    VOLUME_PER_TIME_PER_LENGTH = "volume per time per length"
    VOLUME_PER_TIME_PER_TIME = "volume per time per time"
    VOLUME_PER_TIME_PER_VOLUME = "volume per time per volume"
    VOLUME_PER_VOLUME = "volume per volume"
    VOLUMETRIC_HEAT_TRANSFER_COEFFICIENT = "volumetric heat transfer coefficient"
    VOLUMETRIC_THERMAL_EXPANSION = "volumetric thermal expansion"
    WORK = "work"
    YOUNG_MODULUS = "Young modulus"


class ResqmlUom(Enum):
    VALUE = "%"
    AREA = "%[area]"
    MASS = "%[mass]"
    MOLAR = "%[molar]"
    VOL = "%[vol]"
    BBL_D_BBL_D = "(bbl/d)/(bbl/d)"
    M3_D_M3_D = "(m3/d)/(m3/d)"
    M3_S_M3_S = "(m3/s)/(m3/s)"
    VALUE_0_001_BBL_FT3 = "0.001 bbl/ft3"
    VALUE_0_001_BBL_M3 = "0.001 bbl/m3"
    VALUE_0_001_D_FT3 = "0.001 d/ft3"
    VALUE_0_001_GAL_UK_BBL = "0.001 gal[UK]/bbl"
    VALUE_0_001_GAL_UK_GAL_UK = "0.001 gal[UK]/gal[UK]"
    VALUE_0_001_GAL_US_BBL = "0.001 gal[US]/bbl"
    VALUE_0_001_GAL_US_FT3 = "0.001 gal[US]/ft3"
    VALUE_0_001_GAL_US_GAL_US = "0.001 gal[US]/gal[US]"
    VALUE_0_001_H_FT = "0.001 h/ft"
    VALUE_0_001_K_PA2_C_P = "0.001 kPa2/cP"
    VALUE_0_001_LBM_BBL = "0.001 lbm/bbl"
    VALUE_0_001_LBM_GAL_UK = "0.001 lbm/gal[UK]"
    VALUE_0_001_LBM_GAL_US = "0.001 lbm/gal[US]"
    VALUE_0_001_PSI_FT = "0.001 psi/ft"
    VALUE_0_001_PT_UK_BBL = "0.001 pt[UK]/bbl"
    VALUE_0_001_SECA = "0.001 seca"
    VALUE_0_01_BBL_BBL = "0.01 bbl/bbl"
    VALUE_0_01_DEGA_FT = "0.01 dega/ft"
    VALUE_0_01_DEG_F_FT = "0.01 degF/ft"
    VALUE_0_01_DM3_KM = "0.01 dm3/km"
    VALUE_0_01_FT_FT = "0.01 ft/ft"
    VALUE_0_01_GRAIN_FT3 = "0.01 grain/ft3"
    VALUE_0_01_L_KG = "0.01 L/kg"
    VALUE_0_01_L_KM = "0.01 L/km"
    VALUE_0_01_LBF_FT = "0.01 lbf/ft"
    VALUE_0_01_LBF_FT2 = "0.01 lbf/ft2"
    VALUE_0_01_LBM_FT2 = "0.01 lbm/ft2"
    VALUE_0_01_PSI_FT = "0.01 psi/ft"
    VALUE_0_1_FT = "0.1 ft"
    VALUE_0_1_FT_US = "0.1 ft[US]"
    VALUE_0_1_GAL_US_BBL = "0.1 gal[US]/bbl"
    VALUE_0_1_IN = "0.1 in"
    VALUE_0_1_L_BBL = "0.1 L/bbl"
    VALUE_0_1_LBM_BBL = "0.1 lbm/bbl"
    VALUE_0_1_PT_US_BBL = "0.1 pt[US]/bbl"
    VALUE_0_1_YD = "0.1 yd"
    VALUE_1_KG_S = "1/(kg.s)"
    VALUE_1_16_IN = "1/16 in"
    VALUE_1_2_FT = "1/2 ft"
    VALUE_1_2_MS = "1/2 ms"
    VALUE_1_30_CM3_MIN = "1/30 cm3/min"
    VALUE_1_30_DEGA_FT = "1/30 dega/ft"
    VALUE_1_30_DEGA_M = "1/30 dega/m"
    VALUE_1_30_LBF_M = "1/30 lbf/m"
    VALUE_1_30_M_M = "1/30 m/m"
    VALUE_1_30_N_M = "1/30 N/m"
    VALUE_1_32_IN = "1/32 in"
    VALUE_1_64_IN = "1/64 in"
    VALUE_1_A = "1/a"
    VALUE_1_ANGSTROM = "1/angstrom"
    VALUE_1_BAR = "1/bar"
    VALUE_1_BBL = "1/bbl"
    VALUE_1_CM = "1/cm"
    VALUE_1_D = "1/d"
    VALUE_1_DEG_C = "1/degC"
    VALUE_1_DEG_F = "1/degF"
    VALUE_1_DEG_R = "1/degR"
    VALUE_1_FT = "1/ft"
    VALUE_1_FT2 = "1/ft2"
    VALUE_1_FT3 = "1/ft3"
    VALUE_1_G = "1/g"
    VALUE_1_GAL_UK = "1/gal[UK]"
    VALUE_1_GAL_US = "1/gal[US]"
    VALUE_1_H = "1/H"
    VALUE_1_H_1 = "1/h"
    VALUE_1_IN = "1/in"
    VALUE_1_K = "1/K"
    VALUE_1_KG = "1/kg"
    VALUE_1_KM2 = "1/km2"
    VALUE_1_K_PA = "1/kPa"
    VALUE_1_L = "1/L"
    VALUE_1_LBF = "1/lbf"
    VALUE_1_LBM = "1/lbm"
    VALUE_1_M = "1/m"
    VALUE_1_M2 = "1/m2"
    VALUE_1_M3 = "1/m3"
    VALUE_1_MI = "1/mi"
    VALUE_1_MI2 = "1/mi2"
    VALUE_1_MIN = "1/min"
    VALUE_1_MM = "1/mm"
    VALUE_1_MS = "1/ms"
    VALUE_1_N = "1/N"
    VALUE_1_NM = "1/nm"
    VALUE_1_PA = "1/Pa"
    VALUE_1_P_PA = "1/pPa"
    VALUE_1_PSI = "1/psi"
    VALUE_1_S = "1/s"
    VALUE_1_UPSI = "1/upsi"
    VALUE_1_US = "1/us"
    VALUE_1_U_V = "1/uV"
    VALUE_1_V = "1/V"
    VALUE_1_WK = "1/wk"
    VALUE_1_YD = "1/yd"
    VALUE_10_FT = "10 ft"
    VALUE_10_IN = "10 in"
    VALUE_10_KM = "10 km"
    VALUE_10_K_N = "10 kN"
    VALUE_10_MG_M3 = "10 Mg/m3"
    VALUE_100_FT = "100 ft"
    VALUE_100_KA_T = "100 ka[t]"
    VALUE_100_KM = "100 km"
    VALUE_1000_BBL = "1000 bbl"
    VALUE_1000_BBL_FT_D = "1000 bbl.ft/d"
    VALUE_1000_BBL_D = "1000 bbl/d"
    VALUE_1000_FT = "1000 ft"
    VALUE_1000_FT_H = "1000 ft/h"
    VALUE_1000_FT_S = "1000 ft/s"
    VALUE_1000_FT3 = "1000 ft3"
    VALUE_1000_FT3_D_FT = "1000 ft3/(d.ft)"
    VALUE_1000_FT3_PSI_D = "1000 ft3/(psi.d)"
    VALUE_1000_FT3_BBL = "1000 ft3/bbl"
    VALUE_1000_FT3_D = "1000 ft3/d"
    VALUE_1000_GAL_UK = "1000 gal[UK]"
    VALUE_1000_GAL_US = "1000 gal[US]"
    VALUE_1000_LBF_FT = "1000 lbf.ft"
    VALUE_1000_M3 = "1000 m3"
    VALUE_1000_M3_D_M = "1000 m3/(d.m)"
    VALUE_1000_M3_H_M = "1000 m3/(h.m)"
    VALUE_1000_M3_D = "1000 m3/d"
    VALUE_1000_M3_H = "1000 m3/h"
    VALUE_1000_M3_M3 = "1000 m3/m3"
    VALUE_1000_M4_D = "1000 m4/d"
    VALUE_1_E_6_ACRE_FT_BBL = "1E-6 acre.ft/bbl"
    VALUE_1_E_6_BBL_FT3 = "1E-6 bbl/ft3"
    VALUE_1_E_6_BBL_M3 = "1E-6 bbl/m3"
    VALUE_1_E_6_GAL_US = "1E-6 gal[US]"
    VALUE_1_E_6_M3_M3_DEG_C = "1E-6 m3/(m3.degC)"
    VALUE_1_E_6_M3_M3_DEG_F = "1E-6 m3/(m3.degF)"
    VALUE_1_E_9_1_FT = "1E-9 1/ft"
    VALUE_1_E12_FT3 = "1E12 ft3"
    VALUE_1_E6_FT3_D_BBL_D = "1E6 (ft3/d)/(bbl/d)"
    VALUE_1_E6_BBL = "1E6 bbl"
    VALUE_1_E6_BBL_ACRE_FT = "1E6 bbl/(acre.ft)"
    VALUE_1_E6_BBL_ACRE = "1E6 bbl/acre"
    VALUE_1_E6_BBL_D = "1E6 bbl/d"
    VALUE_1_E6_BTU_IT = "1E6 Btu[IT]"
    VALUE_1_E6_BTU_IT_H = "1E6 Btu[IT]/h"
    VALUE_1_E6_FT3 = "1E6 ft3"
    VALUE_1_E6_FT3_ACRE_FT = "1E6 ft3/(acre.ft)"
    VALUE_1_E6_FT3_BBL = "1E6 ft3/bbl"
    VALUE_1_E6_FT3_D = "1E6 ft3/d"
    VALUE_1_E6_LBM_A = "1E6 lbm/a"
    VALUE_1_E6_M3 = "1E6 m3"
    VALUE_1_E6_M3_D = "1E6 m3/d"
    VALUE_1_E9_BBL = "1E9 bbl"
    VALUE_1_E9_FT3 = "1E9 ft3"
    VALUE_30_FT = "30 ft"
    VALUE_30_M = "30 m"
    A = "A"
    A_1 = "a"
    A_H = "A.h"
    A_M2 = "A.m2"
    A_S = "A.s"
    A_S_KG = "A.s/kg"
    A_S_M3 = "A.s/m3"
    A_CM2 = "A/cm2"
    A_FT2 = "A/ft2"
    A_M = "A/m"
    A_M2_1 = "A/m2"
    A_MM = "A/mm"
    A_MM2 = "A/mm2"
    A_T = "a[t]"
    ACRE = "acre"
    ACRE_FT = "acre.ft"
    AG = "ag"
    A_J = "aJ"
    ANGSTROM = "angstrom"
    AT_1 = "at"
    ATM = "atm"
    ATM_FT = "atm/ft"
    ATM_H = "atm/h"
    ATM_HM = "atm/hm"
    ATM_M = "atm/m"
    B = "b"
    B_1 = "B"
    B_W = "B.W"
    B_CM3 = "b/cm3"
    B_M = "B/m"
    B_O = "B/O"
    BAR = "bar"
    BAR_H = "bar/h"
    BAR_KM = "bar/km"
    BAR_M = "bar/m"
    BAR2 = "bar2"
    BAR2_C_P = "bar2/cP"
    BBL = "bbl"
    BBL_ACRE_FT = "bbl/(acre.ft)"
    BBL_D_ACRE_FT = "bbl/(d.acre.ft)"
    BBL_D_FT = "bbl/(d.ft)"
    BBL_FT_PSI_D = "bbl/(ft.psi.d)"
    BBL_K_PA_D = "bbl/(kPa.d)"
    BBL_PSI_D = "bbl/(psi.d)"
    BBL_ACRE = "bbl/acre"
    BBL_BBL = "bbl/bbl"
    BBL_D = "bbl/d"
    BBL_D2 = "bbl/d2"
    BBL_FT = "bbl/ft"
    BBL_FT3 = "bbl/ft3"
    BBL_H = "bbl/h"
    BBL_H2 = "bbl/h2"
    BBL_IN = "bbl/in"
    BBL_M3 = "bbl/m3"
    BBL_MI = "bbl/mi"
    BBL_MIN = "bbl/min"
    BBL_PSI = "bbl/psi"
    BBL_TON_UK = "bbl/ton[UK]"
    BBL_TON_US = "bbl/ton[US]"
    BD = "Bd"
    BIT = "bit"
    BIT_S = "bit/s"
    BQ = "Bq"
    BQ_KG = "Bq/kg"
    BTU_IT = "Btu[IT]"
    BTU_IT_IN_H_FT2_DEG_F = "Btu[IT].in/(h.ft2.degF)"
    BTU_IT_H_FT_DEG_F = "Btu[IT]/(h.ft.degF)"
    BTU_IT_H_FT2 = "Btu[IT]/(h.ft2)"
    BTU_IT_H_FT2_DEG_F = "Btu[IT]/(h.ft2.degF)"
    BTU_IT_H_FT2_DEG_R = "Btu[IT]/(h.ft2.degR)"
    BTU_IT_H_FT3 = "Btu[IT]/(h.ft3)"
    BTU_IT_H_FT3_DEG_F = "Btu[IT]/(h.ft3.degF)"
    BTU_IT_H_M2_DEG_C = "Btu[IT]/(h.m2.degC)"
    BTU_IT_HP_H = "Btu[IT]/(hp.h)"
    BTU_IT_LBM_DEG_F = "Btu[IT]/(lbm.degF)"
    BTU_IT_LBM_DEG_R = "Btu[IT]/(lbm.degR)"
    BTU_IT_LBMOL_DEG_F = "Btu[IT]/(lbmol.degF)"
    BTU_IT_S_FT2 = "Btu[IT]/(s.ft2)"
    BTU_IT_S_FT2_DEG_F = "Btu[IT]/(s.ft2.degF)"
    BTU_IT_S_FT3 = "Btu[IT]/(s.ft3)"
    BTU_IT_S_FT3_DEG_F = "Btu[IT]/(s.ft3.degF)"
    BTU_IT_BBL = "Btu[IT]/bbl"
    BTU_IT_FT3 = "Btu[IT]/ft3"
    BTU_IT_GAL_UK = "Btu[IT]/gal[UK]"
    BTU_IT_GAL_US = "Btu[IT]/gal[US]"
    BTU_IT_H = "Btu[IT]/h"
    BTU_IT_LBM = "Btu[IT]/lbm"
    BTU_IT_LBMOL = "Btu[IT]/lbmol"
    BTU_IT_MIN = "Btu[IT]/min"
    BTU_IT_S = "Btu[IT]/s"
    BTU_TH = "Btu[th]"
    BTU_UK = "Btu[UK]"
    BYTE = "byte"
    BYTE_S = "byte/s"
    C = "C"
    C_M = "C.m"
    C_CM2 = "C/cm2"
    C_CM3 = "C/cm3"
    C_G = "C/g"
    C_KG = "C/kg"
    C_M2 = "C/m2"
    C_M3 = "C/m3"
    C_MM2 = "C/mm2"
    C_MM3 = "C/mm3"
    CA = "ca"
    C_A_1 = "cA"
    CAL_IT = "cal[IT]"
    CAL_TH = "cal[th]"
    CAL_TH_G_K = "cal[th]/(g.K)"
    CAL_TH_H_CM_DEG_C = "cal[th]/(h.cm.degC)"
    CAL_TH_H_CM2 = "cal[th]/(h.cm2)"
    CAL_TH_H_CM2_DEG_C = "cal[th]/(h.cm2.degC)"
    CAL_TH_H_CM3 = "cal[th]/(h.cm3)"
    CAL_TH_MOL_DEG_C = "cal[th]/(mol.degC)"
    CAL_TH_S_CM_DEG_C = "cal[th]/(s.cm.degC)"
    CAL_TH_S_CM2_DEG_C = "cal[th]/(s.cm2.degC)"
    CAL_TH_S_CM3 = "cal[th]/(s.cm3)"
    CAL_TH_CM3 = "cal[th]/cm3"
    CAL_TH_G = "cal[th]/g"
    CAL_TH_H = "cal[th]/h"
    CAL_TH_KG = "cal[th]/kg"
    CAL_TH_LBM = "cal[th]/lbm"
    CAL_TH_M_L = "cal[th]/mL"
    CAL_TH_MM3 = "cal[th]/mm3"
    C_C = "cC"
    CCAL_TH = "ccal[th]"
    CCGR = "ccgr"
    CD = "cd"
    CD_M2 = "cd/m2"
    C_EUC = "cEuc"
    CE_V = "ceV"
    C_F = "cF"
    CG_1 = "cg"
    CGAUSS = "cgauss"
    CGR = "cgr"
    C_GY = "cGy"
    C_H = "cH"
    CHAIN = "chain"
    CHAIN_BN_A = "chain[BnA]"
    CHAIN_BN_B = "chain[BnB]"
    CHAIN_CLA = "chain[Cla]"
    CHAIN_IND37 = "chain[Ind37]"
    CHAIN_SE = "chain[Se]"
    CHAIN_SE_T = "chain[SeT]"
    CHAIN_US = "chain[US]"
    C_HZ = "cHz"
    CI = "Ci"
    C_J = "cJ"
    CM_1 = "cm"
    CM_A = "cm/a"
    CM_S = "cm/s"
    CM_S2 = "cm/s2"
    CM2_1 = "cm2"
    CM2_G = "cm2/g"
    CM2_S = "cm2/s"
    CM3_1 = "cm3"
    CM3_CM3 = "cm3/cm3"
    CM3_G = "cm3/g"
    CM3_H = "cm3/h"
    CM3_L = "cm3/L"
    CM3_M3 = "cm3/m3"
    CM3_MIN = "cm3/min"
    CM3_S = "cm3/s"
    CM4 = "cm4"
    CM_H2_O_4DEG_C = "cmH2O[4degC]"
    C_N = "cN"
    COHM = "cohm"
    C_P = "cP"
    C_PA = "cPa"
    CRD = "crd"
    C_S = "cS"
    CS_1 = "cs"
    C_ST = "cSt"
    CT = "ct"
    C_T_1 = "cT"
    CU = "cu"
    C_V = "cV"
    C_W = "cW"
    C_WB = "cWb"
    CWT_UK = "cwt[UK]"
    CWT_US = "cwt[US]"
    D = "d"
    D_1 = "D"
    D_FT = "D.ft"
    D_M = "D.m"
    D_PA_S = "D/(Pa.s)"
    D_BBL = "d/bbl"
    D_C_P = "D/cP"
    D_FT3 = "d/ft3"
    D_M3 = "d/m3"
    D_API = "D[API]"
    D_A = "dA"
    DAM = "dam"
    DA_N = "daN"
    DA_N_M = "daN.m"
    D_API_1 = "dAPI"
    D_B = "dB"
    D_B_MW = "dB.MW"
    D_B_M_W_1 = "dB.mW"
    D_B_W = "dB.W"
    D_B_FT = "dB/ft"
    D_B_KM = "dB/km"
    D_B_M = "dB/m"
    D_B_O = "dB/O"
    D_C = "dC"
    DCAL_TH = "dcal[th]"
    DEGA = "dega"
    DEGA_FT = "dega/ft"
    DEGA_H = "dega/h"
    DEGA_M = "dega/m"
    DEGA_MIN = "dega/min"
    DEGA_S = "dega/s"
    DEG_C = "degC"
    DEG_C_M2_H_KCAL_TH = "degC.m2.h/kcal[th]"
    DEG_C_FT = "degC/ft"
    DEG_C_H = "degC/h"
    DEG_C_HM = "degC/hm"
    DEG_C_KM = "degC/km"
    DEG_C_K_PA = "degC/kPa"
    DEG_C_M = "degC/m"
    DEG_C_MIN = "degC/min"
    DEG_C_S = "degC/s"
    DEG_F = "degF"
    DEG_F_FT2_H_BTU_IT = "degF.ft2.h/Btu[IT]"
    DEG_F_FT = "degF/ft"
    DEG_F_H = "degF/h"
    DEG_F_M = "degF/m"
    DEG_F_MIN = "degF/min"
    DEG_F_PSI = "degF/psi"
    DEG_F_S = "degF/s"
    DEG_R = "degR"
    D_EUC = "dEuc"
    DE_V = "deV"
    D_F = "dF"
    DGAUSS = "dgauss"
    D_GY = "dGy"
    D_H = "dH"
    D_HZ = "dHz"
    D_J = "dJ"
    DM_1 = "dm"
    DM_S = "dm/s"
    DM3_1 = "dm3"
    DM3_K_W_H = "dm3/(kW.h)"
    DM3_KG = "dm3/kg"
    DM3_KMOL = "dm3/kmol"
    DM3_M = "dm3/m"
    DM3_M3 = "dm3/m3"
    DM3_MJ = "dm3/MJ"
    DM3_S = "dm3/s"
    DM3_S2 = "dm3/s2"
    DM3_T = "dm3/t"
    D_N = "dN"
    D_N_M = "dN.m"
    DOHM = "dohm"
    D_P = "dP"
    D_PA = "dPa"
    DRD = "drd"
    DS = "ds"
    D_S_1 = "dS"
    D_T = "dT"
    D_V = "dV"
    D_W = "dW"
    D_WB = "dWb"
    DYNE = "dyne"
    DYNE_CM2 = "dyne.cm2"
    DYNE_S_CM2 = "dyne.s/cm2"
    DYNE_CM = "dyne/cm"
    DYNE_CM2_1 = "dyne/cm2"
    EA = "EA"
    EA_T = "Ea[t]"
    EC = "EC"
    ECAL_TH = "Ecal[th]"
    EEUC = "EEuc"
    EE_V = "EeV"
    EF = "EF"
    EG = "Eg"
    EGAUSS = "Egauss"
    EGY = "EGy"
    EH = "EH"
    EHZ = "EHz"
    EJ = "EJ"
    EJ_A = "EJ/a"
    EM = "Em"
    EN = "EN"
    EOHM = "Eohm"
    EP = "EP"
    EPA = "EPa"
    ERD = "Erd"
    ERG = "erg"
    ERG_A = "erg/a"
    ERG_CM2 = "erg/cm2"
    ERG_CM3 = "erg/cm3"
    ERG_G = "erg/g"
    ERG_KG = "erg/kg"
    ERG_M3 = "erg/m3"
    ES = "ES"
    ET = "ET"
    EUC = "Euc"
    E_V = "eV"
    EW = "EW"
    EWB = "EWb"
    F = "F"
    F_M = "F/m"
    FA = "fa"
    F_A_1 = "fA"
    FATHOM = "fathom"
    F_C = "fC"
    FCAL_TH = "fcal[th]"
    F_EUC = "fEuc"
    FE_V = "feV"
    F_F = "fF"
    FG = "fg"
    FGAUSS = "fgauss"
    F_GY = "fGy"
    F_H = "fH"
    F_HZ = "fHz"
    F_J = "fJ"
    FLOZ_UK = "floz[UK]"
    FLOZ_US = "floz[US]"
    FM_1 = "fm"
    F_N = "fN"
    FOHM = "fohm"
    FOOTCANDLE = "footcandle"
    FOOTCANDLE_S = "footcandle.s"
    F_P = "fP"
    F_PA = "fPa"
    FRD = "frd"
    F_S = "fS"
    FT = "ft"
    F_T_1 = "fT"
    FT_BBL = "ft/bbl"
    FT_D = "ft/d"
    FT_DEG_F = "ft/degF"
    FT_FT = "ft/ft"
    FT_FT3 = "ft/ft3"
    FT_GAL_US = "ft/gal[US]"
    FT_H = "ft/h"
    FT_IN = "ft/in"
    FT_LBM = "ft/lbm"
    FT_M = "ft/m"
    FT_MI = "ft/mi"
    FT_MIN = "ft/min"
    FT_MS = "ft/ms"
    FT_PSI = "ft/psi"
    FT_S = "ft/s"
    FT_S2 = "ft/s2"
    FT_US = "ft/us"
    FT_BN_A = "ft[BnA]"
    FT_BN_B = "ft[BnB]"
    FT_BR36 = "ft[Br36]"
    FT_BR65 = "ft[Br65]"
    FT_CLA = "ft[Cla]"
    FT_GC = "ft[GC]"
    FT_IND = "ft[Ind]"
    FT_IND37 = "ft[Ind37]"
    FT_IND62 = "ft[Ind62]"
    FT_IND75 = "ft[Ind75]"
    FT_SE = "ft[Se]"
    FT_SE_T = "ft[SeT]"
    FT_US_1 = "ft[US]"
    FT2 = "ft2"
    FT2_H = "ft2/h"
    FT2_IN3 = "ft2/in3"
    FT2_LBM = "ft2/lbm"
    FT2_S = "ft2/s"
    FT3 = "ft3"
    FT3_D_FT = "ft3/(d.ft)"
    FT3_FT_PSI_D = "ft3/(ft.psi.d)"
    FT3_MIN_FT2 = "ft3/(min.ft2)"
    FT3_S_FT2 = "ft3/(s.ft2)"
    FT3_BBL = "ft3/bbl"
    FT3_D = "ft3/d"
    FT3_D2 = "ft3/d2"
    FT3_FT = "ft3/ft"
    FT3_FT2 = "ft3/ft2"
    FT3_FT3 = "ft3/ft3"
    FT3_H = "ft3/h"
    FT3_H2 = "ft3/h2"
    FT3_KG = "ft3/kg"
    FT3_LBM = "ft3/lbm"
    FT3_LBMOL = "ft3/lbmol"
    FT3_MIN = "ft3/min"
    FT3_MIN2 = "ft3/min2"
    FT3_RAD = "ft3/rad"
    FT3_S = "ft3/s"
    FT3_S2 = "ft3/s2"
    FT3_SACK_94LBM = "ft3/sack[94lbm]"
    FUR_US = "fur[US]"
    F_V = "fV"
    F_W = "fW"
    F_WB = "fWb"
    G = "g"
    G_FT_CM3_S = "g.ft/(cm3.s)"
    G_M_CM3_S = "g.m/(cm3.s)"
    G_CM3 = "g/cm3"
    G_CM4 = "g/cm4"
    G_DM3 = "g/dm3"
    G_GAL_UK = "g/gal[UK]"
    G_GAL_US = "g/gal[US]"
    G_KG = "g/kg"
    G_L = "g/L"
    G_M3 = "g/m3"
    G_MOL = "g/mol"
    G_S = "g/s"
    G_T = "g/t"
    GA = "GA"
    GA_T = "Ga[t]"
    GAL = "Gal"
    GAL_UK = "gal[UK]"
    GAL_UK_H_FT = "gal[UK]/(h.ft)"
    GAL_UK_H_FT2 = "gal[UK]/(h.ft2)"
    GAL_UK_H_IN = "gal[UK]/(h.in)"
    GAL_UK_H_IN2 = "gal[UK]/(h.in2)"
    GAL_UK_MIN_FT = "gal[UK]/(min.ft)"
    GAL_UK_MIN_FT2 = "gal[UK]/(min.ft2)"
    GAL_UK_D = "gal[UK]/d"
    GAL_UK_FT3 = "gal[UK]/ft3"
    GAL_UK_H = "gal[UK]/h"
    GAL_UK_H2 = "gal[UK]/h2"
    GAL_UK_LBM = "gal[UK]/lbm"
    GAL_UK_MI = "gal[UK]/mi"
    GAL_UK_MIN = "gal[UK]/min"
    GAL_UK_MIN2 = "gal[UK]/min2"
    GAL_US = "gal[US]"
    GAL_US_H_FT = "gal[US]/(h.ft)"
    GAL_US_H_FT2 = "gal[US]/(h.ft2)"
    GAL_US_H_IN = "gal[US]/(h.in)"
    GAL_US_H_IN2 = "gal[US]/(h.in2)"
    GAL_US_MIN_FT = "gal[US]/(min.ft)"
    GAL_US_MIN_FT2 = "gal[US]/(min.ft2)"
    GAL_US_BBL = "gal[US]/bbl"
    GAL_US_D = "gal[US]/d"
    GAL_US_FT = "gal[US]/ft"
    GAL_US_FT3 = "gal[US]/ft3"
    GAL_US_H = "gal[US]/h"
    GAL_US_H2 = "gal[US]/h2"
    GAL_US_LBM = "gal[US]/lbm"
    GAL_US_MI = "gal[US]/mi"
    GAL_US_MIN = "gal[US]/min"
    GAL_US_MIN2 = "gal[US]/min2"
    GAL_US_SACK_94LBM = "gal[US]/sack[94lbm]"
    GAL_US_TON_UK = "gal[US]/ton[UK]"
    GAL_US_TON_US = "gal[US]/ton[US]"
    G_API = "gAPI"
    GAUSS = "gauss"
    GAUSS_CM = "gauss/cm"
    GBQ = "GBq"
    GC = "GC"
    GCAL_TH = "Gcal[th]"
    GEUC = "GEuc"
    GE_V = "GeV"
    GF = "gf"
    GF_1 = "GF"
    GG = "Gg"
    GGAUSS = "Ggauss"
    GGY = "GGy"
    GH = "GH"
    GHZ = "GHz"
    GJ = "GJ"
    GM = "Gm"
    GN = "gn"
    GN_1 = "GN"
    GOHM = "Gohm"
    GON = "gon"
    GP = "GP"
    GPA = "GPa"
    GPA_CM = "GPa/cm"
    GPA2 = "GPa2"
    GRAIN = "grain"
    GRAIN_FT3 = "grain/ft3"
    GRAIN_GAL_US = "grain/gal[US]"
    GRD = "Grd"
    GS_1 = "GS"
    GT_1 = "GT"
    GV = "GV"
    GW = "GW"
    GW_H = "GW.h"
    GWB = "GWb"
    GY = "Gy"
    H = "H"
    H_1 = "h"
    H_FT3 = "h/ft3"
    H_KM = "h/km"
    H_M = "H/m"
    H_M3 = "h/m3"
    HA = "ha"
    HA_M = "ha.m"
    HBAR = "hbar"
    HG = "hg"
    H_L = "hL"
    HM_1 = "hm"
    H_N = "hN"
    HP = "hp"
    HP_H = "hp.h"
    HP_H_BBL = "hp.h/bbl"
    HP_H_LBM = "hp.h/lbm"
    HP_FT3 = "hp/ft3"
    HP_IN2 = "hp/in2"
    HP_ELEC = "hp[elec]"
    HP_HYD = "hp[hyd]"
    HP_HYD_IN2 = "hp[hyd]/in2"
    HP_METRIC = "hp[metric]"
    HP_METRIC_H = "hp[metric].h"
    HS = "hs"
    HZ = "Hz"
    IN = "in"
    IN_IN_DEG_F = "in/(in.degF)"
    IN_A = "in/a"
    IN_MIN = "in/min"
    IN_S = "in/s"
    IN_S2 = "in/s2"
    IN_US = "in[US]"
    IN2 = "in2"
    IN2_FT2 = "in2/ft2"
    IN2_IN2 = "in2/in2"
    IN2_S = "in2/s"
    IN3 = "in3"
    IN3_FT = "in3/ft"
    IN4 = "in4"
    IN_H2_O_39DEG_F = "inH2O[39degF]"
    IN_H2_O_60DEG_F = "inH2O[60degF]"
    IN_HG_32DEG_F = "inHg[32degF]"
    IN_HG_60DEG_F = "inHg[60degF]"
    J = "J"
    J_M_S_M2_K = "J.m/(s.m2.K)"
    J_M_M2 = "J.m/m2"
    J_G_K = "J/(g.K)"
    J_KG_K = "J/(kg.K)"
    J_MOL_K = "J/(mol.K)"
    J_S_M2_DEG_C = "J/(s.m2.degC)"
    J_CM2 = "J/cm2"
    J_DM3 = "J/dm3"
    J_G = "J/g"
    J_K = "J/K"
    J_KG = "J/kg"
    J_M = "J/m"
    J_M2 = "J/m2"
    J_M3 = "J/m3"
    J_MOL = "J/mol"
    J_S = "J/s"
    K = "K"
    K_M2_K_W = "K.m2/kW"
    K_M2_W = "K.m2/W"
    K_KM = "K/km"
    K_M = "K/m"
    K_PA = "K/Pa"
    K_S = "K/s"
    K_W = "K/W"
    K_A = "kA"
    KA_T = "ka[t]"
    K_C = "kC"
    KCAL_TH = "kcal[th]"
    KCAL_TH_M_CM2 = "kcal[th].m/cm2"
    KCAL_TH_H_M_DEG_C = "kcal[th]/(h.m.degC)"
    KCAL_TH_H_M2_DEG_C = "kcal[th]/(h.m2.degC)"
    KCAL_TH_KG_DEG_C = "kcal[th]/(kg.degC)"
    KCAL_TH_CM3 = "kcal[th]/cm3"
    KCAL_TH_G = "kcal[th]/g"
    KCAL_TH_H = "kcal[th]/h"
    KCAL_TH_KG = "kcal[th]/kg"
    KCAL_TH_M3 = "kcal[th]/m3"
    KCAL_TH_MOL = "kcal[th]/mol"
    KCD = "kcd"
    KDYNE = "kdyne"
    K_EUC = "kEuc"
    KE_V = "keV"
    K_F = "kF"
    KG = "kg"
    KG_M = "kg.m"
    KG_M_CM2 = "kg.m/cm2"
    KG_M_S = "kg.m/s"
    KG_M2 = "kg.m2"
    KG_K_W_H = "kg/(kW.h)"
    KG_M_S_1 = "kg/(m.s)"
    KG_M2_S = "kg/(m2.s)"
    KG_D = "kg/d"
    KG_DM3 = "kg/dm3"
    KG_DM4 = "kg/dm4"
    KG_H = "kg/h"
    KG_J = "kg/J"
    KG_KG = "kg/kg"
    KG_L = "kg/L"
    KG_M_1 = "kg/m"
    KG_M2_1 = "kg/m2"
    KG_M3 = "kg/m3"
    KG_M4 = "kg/m4"
    KG_MIN = "kg/min"
    KG_MJ = "kg/MJ"
    KG_MOL = "kg/mol"
    KG_S = "kg/s"
    KG_SACK_94LBM = "kg/sack[94lbm]"
    KG_T = "kg/t"
    KGAUSS = "kgauss"
    KGF = "kgf"
    KGF_M = "kgf.m"
    KGF_M_CM2 = "kgf.m/cm2"
    KGF_M_M = "kgf.m/m"
    KGF_M2 = "kgf.m2"
    KGF_S_M2 = "kgf.s/m2"
    KGF_CM = "kgf/cm"
    KGF_CM2 = "kgf/cm2"
    KGF_KGF = "kgf/kgf"
    KGF_M2_1 = "kgf/m2"
    KGF_MM2 = "kgf/mm2"
    K_GY = "kGy"
    K_H = "kH"
    K_HZ = "kHz"
    KIBYTE = "Kibyte"
    K_J = "kJ"
    K_J_M_H_M2_K = "kJ.m/(h.m2.K)"
    K_J_H_M2_K = "kJ/(h.m2.K)"
    K_J_KG_K = "kJ/(kg.K)"
    K_J_KMOL_K = "kJ/(kmol.K)"
    K_J_DM3 = "kJ/dm3"
    K_J_KG = "kJ/kg"
    K_J_KMOL = "kJ/kmol"
    K_J_M3 = "kJ/m3"
    KLBF = "klbf"
    KLBM = "klbm"
    KLBM_IN = "klbm/in"
    KLX = "klx"
    KM_1 = "km"
    KM_CM = "km/cm"
    KM_DM3 = "km/dm3"
    KM_H = "km/h"
    KM_L = "km/L"
    KM_S = "km/s"
    KM2 = "km2"
    KM3 = "km3"
    KMOL = "kmol"
    KMOL_H = "kmol/h"
    KMOL_M3 = "kmol/m3"
    KMOL_S = "kmol/s"
    K_N = "kN"
    K_N_M = "kN.m"
    K_N_M2 = "kN.m2"
    K_N_M_1 = "kN/m"
    K_N_M2_1 = "kN/m2"
    KNOT = "knot"
    KOHM = "kohm"
    KOHM_M = "kohm.m"
    K_P = "kP"
    K_PA_1 = "kPa"
    K_PA_S_M = "kPa.s/m"
    K_PA_H = "kPa/h"
    K_PA_HM = "kPa/hm"
    K_PA_M = "kPa/m"
    K_PA_MIN = "kPa/min"
    K_PA2 = "kPa2"
    K_PA2_C_P = "kPa2/cP"
    KPSI = "kpsi"
    KPSI2 = "kpsi2"
    KRAD = "krad"
    KRD = "krd"
    K_S_1 = "kS"
    K_S_M = "kS/m"
    K_T = "kT"
    K_V = "kV"
    K_W_1 = "kW"
    K_W_H = "kW.h"
    K_W_H_KG_DEG_C = "kW.h/(kg.degC)"
    K_W_H_DM3 = "kW.h/dm3"
    K_W_H_KG = "kW.h/kg"
    K_W_H_M3 = "kW.h/m3"
    K_W_M2_K = "kW/(m2.K)"
    K_W_M3_K = "kW/(m3.K)"
    K_W_CM2 = "kW/cm2"
    K_W_M2 = "kW/m2"
    K_W_M3 = "kW/m3"
    K_WB = "kWb"
    L = "L"
    L_BAR_MIN = "L/(bar.min)"
    L_H = "L/h"
    L_KG = "L/kg"
    L_KMOL = "L/kmol"
    L_M = "L/m"
    L_M3 = "L/m3"
    L_MIN = "L/min"
    L_MOL = "L/mol"
    L_S = "L/s"
    L_S2 = "L/s2"
    L_T = "L/t"
    L_TON_UK = "L/ton[UK]"
    LBF = "lbf"
    LBF_FT = "lbf.ft"
    LBF_FT_BBL = "lbf.ft/bbl"
    LBF_FT_GAL_US = "lbf.ft/gal[US]"
    LBF_FT_IN = "lbf.ft/in"
    LBF_FT_IN2 = "lbf.ft/in2"
    LBF_FT_LBM = "lbf.ft/lbm"
    LBF_FT_MIN = "lbf.ft/min"
    LBF_FT_S = "lbf.ft/s"
    LBF_IN = "lbf.in"
    LBF_IN_IN = "lbf.in/in"
    LBF_IN2 = "lbf.in2"
    LBF_S_FT2 = "lbf.s/ft2"
    LBF_S_IN2 = "lbf.s/in2"
    LBF_FT_1 = "lbf/ft"
    LBF_FT2 = "lbf/ft2"
    LBF_FT3 = "lbf/ft3"
    LBF_GAL_US = "lbf/gal[US]"
    LBF_IN_1 = "lbf/in"
    LBF_LBF = "lbf/lbf"
    LBM = "lbm"
    LBM_FT = "lbm.ft"
    LBM_FT_S = "lbm.ft/s"
    LBM_FT2 = "lbm.ft2"
    LBM_FT2_S2 = "lbm.ft2/s2"
    LBM_FT_H = "lbm/(ft.h)"
    LBM_FT_S_1 = "lbm/(ft.s)"
    LBM_FT2_H = "lbm/(ft2.h)"
    LBM_FT2_S = "lbm/(ft2.s)"
    LBM_GAL_UK_FT = "lbm/(gal[UK].ft)"
    LBM_GAL_US_FT = "lbm/(gal[US].ft)"
    LBM_HP_H = "lbm/(hp.h)"
    LBM_BBL = "lbm/bbl"
    LBM_D = "lbm/d"
    LBM_FT_1 = "lbm/ft"
    LBM_FT2_1 = "lbm/ft2"
    LBM_FT3 = "lbm/ft3"
    LBM_FT4 = "lbm/ft4"
    LBM_GAL_UK = "lbm/gal[UK]"
    LBM_GAL_US = "lbm/gal[US]"
    LBM_H = "lbm/h"
    LBM_IN3 = "lbm/in3"
    LBM_LBMOL = "lbm/lbmol"
    LBM_MIN = "lbm/min"
    LBM_S = "lbm/s"
    LBMOL = "lbmol"
    LBMOL_H_FT2 = "lbmol/(h.ft2)"
    LBMOL_S_FT2 = "lbmol/(s.ft2)"
    LBMOL_FT3 = "lbmol/ft3"
    LBMOL_GAL_UK = "lbmol/gal[UK]"
    LBMOL_GAL_US = "lbmol/gal[US]"
    LBMOL_H = "lbmol/h"
    LBMOL_S = "lbmol/s"
    LINK = "link"
    LINK_BN_A = "link[BnA]"
    LINK_BN_B = "link[BnB]"
    LINK_CLA = "link[Cla]"
    LINK_SE = "link[Se]"
    LINK_SE_T = "link[SeT]"
    LINK_US = "link[US]"
    LM_1 = "lm"
    LM_S = "lm.s"
    LM_M2 = "lm/m2"
    LM_W = "lm/W"
    LX = "lx"
    LX_S = "lx.s"
    M = "m"
    M_M_K = "m/(m.K)"
    M_CM = "m/cm"
    M_D = "m/d"
    M_H = "m/h"
    M_K = "m/K"
    M_KG = "m/kg"
    M_KM = "m/km"
    M_K_PA = "m/kPa"
    M_M = "m/m"
    M_M3 = "m/m3"
    M_MIN = "m/min"
    M_MS = "m/ms"
    M_PA = "m/Pa"
    M_S = "m/s"
    M_S2 = "m/s2"
    M_GER = "m[Ger]"
    M2 = "m2"
    M2_K_PA_D = "m2/(kPa.d)"
    M2_PA_S = "m2/(Pa.s)"
    M2_CM3 = "m2/cm3"
    M2_D = "m2/d"
    M2_G = "m2/g"
    M2_H = "m2/h"
    M2_KG = "m2/kg"
    M2_M2 = "m2/m2"
    M2_M3 = "m2/m3"
    M2_MOL = "m2/mol"
    M2_S = "m2/s"
    M3 = "m3"
    M3_BAR_D = "m3/(bar.d)"
    M3_BAR_H = "m3/(bar.h)"
    M3_BAR_MIN = "m3/(bar.min)"
    M3_D_M = "m3/(d.m)"
    M3_H_M = "m3/(h.m)"
    M3_HA_M = "m3/(ha.m)"
    M3_K_PA_D = "m3/(kPa.d)"
    M3_K_PA_H = "m3/(kPa.h)"
    M3_K_W_H = "m3/(kW.h)"
    M3_M3_K = "m3/(m3.K)"
    M3_PA_S = "m3/(Pa.s)"
    M3_PSI_D = "m3/(psi.d)"
    M3_S_FT = "m3/(s.ft)"
    M3_S_M = "m3/(s.m)"
    M3_S_M2 = "m3/(s.m2)"
    M3_S_M3 = "m3/(s.m3)"
    M3_BBL = "m3/bbl"
    M3_D = "m3/d"
    M3_D2 = "m3/d2"
    M3_G = "m3/g"
    M3_H = "m3/h"
    M3_J = "m3/J"
    M3_KG = "m3/kg"
    M3_KM = "m3/km"
    M3_KMOL = "m3/kmol"
    M3_K_PA = "m3/kPa"
    M3_M = "m3/m"
    M3_M2 = "m3/m2"
    M3_M3 = "m3/m3"
    M3_MIN = "m3/min"
    M3_MOL = "m3/mol"
    M3_PA = "m3/Pa"
    M3_RAD = "m3/rad"
    M3_REV = "m3/rev"
    M3_S = "m3/s"
    M3_S2 = "m3/s2"
    M3_T = "m3/t"
    M3_TON_UK = "m3/ton[UK]"
    M3_TON_US = "m3/ton[US]"
    M4 = "m4"
    M4_S = "m4/s"
    M_A = "mA"
    MA_1 = "MA"
    M_A_CM2 = "mA/cm2"
    M_A_FT2 = "mA/ft2"
    MA_T = "Ma[t]"
    MBAR = "mbar"
    MBQ = "MBq"
    M_C = "mC"
    MC_1 = "MC"
    M_C_M2 = "mC/m2"
    MCAL_TH = "Mcal[th]"
    MCAL_TH_1 = "mcal[th]"
    M_CI = "mCi"
    M_D_1 = "mD"
    M_D_FT = "mD.ft"
    M_D_FT2_LBF_S = "mD.ft2/(lbf.s)"
    M_D_IN2_LBF_S = "mD.in2/(lbf.s)"
    M_D_M = "mD.m"
    M_D_PA_S = "mD/(Pa.s)"
    M_D_C_P = "mD/cP"
    MEUC = "MEuc"
    M_EUC_1 = "mEuc"
    ME_V = "meV"
    ME_V_1 = "MeV"
    M_F = "mF"
    MF_1 = "MF"
    MG = "Mg"
    MG_1 = "mg"
    MG_A = "Mg/a"
    MG_D = "Mg/d"
    MG_DM3 = "mg/dm3"
    MG_G = "mg/g"
    MG_GAL_US = "mg/gal[US]"
    MG_H = "Mg/h"
    MG_IN = "Mg/in"
    MG_J = "mg/J"
    MG_KG = "mg/kg"
    MG_L = "mg/L"
    MG_M2 = "Mg/m2"
    MG_M3 = "Mg/m3"
    MG_M3_1 = "mg/m3"
    MG_MIN = "Mg/min"
    M_GAL = "mGal"
    MGAUSS = "Mgauss"
    MGAUSS_1 = "mgauss"
    MGF = "Mgf"
    MGN = "mgn"
    MGY = "MGy"
    M_GY_1 = "mGy"
    MH_1 = "MH"
    M_H_2 = "mH"
    M_HZ = "mHz"
    MHZ_1 = "MHz"
    MI = "mi"
    MI_GAL_UK = "mi/gal[UK]"
    MI_GAL_US = "mi/gal[US]"
    MI_H = "mi/h"
    MI_IN = "mi/in"
    MI_NAUT = "mi[naut]"
    MI_NAUT_UK = "mi[nautUK]"
    MI_US = "mi[US]"
    MI_US_2 = "mi[US]2"
    MI2 = "mi2"
    MI3 = "mi3"
    MIBYTE = "Mibyte"
    MIL = "mil"
    MIL_A = "mil/a"
    MILA_1 = "mila"
    MIN = "min"
    MIN_FT = "min/ft"
    MIN_M = "min/m"
    MINA = "mina"
    M_J = "mJ"
    MJ_1 = "MJ"
    MJ_A = "MJ/a"
    M_J_CM2 = "mJ/cm2"
    MJ_KG = "MJ/kg"
    MJ_KMOL = "MJ/kmol"
    MJ_M = "MJ/m"
    M_J_M2 = "mJ/m2"
    MJ_M3 = "MJ/m3"
    M_L = "mL"
    M_L_GAL_UK = "mL/gal[UK]"
    M_L_GAL_US = "mL/gal[US]"
    M_L_M_L = "mL/mL"
    MM_1 = "Mm"
    MM_4 = "mm"
    MM_MM_K = "mm/(mm.K)"
    MM_A = "mm/a"
    MM_S_1 = "mm/s"
    MM2 = "mm2"
    MM2_MM2 = "mm2/mm2"
    MM2_S = "mm2/s"
    MM3_1 = "mm3"
    MM3_J = "mm3/J"
    MM_HG_0DEG_C = "mmHg[0degC]"
    MMOL = "mmol"
    M_N = "mN"
    MN_1 = "MN"
    M_N_M2 = "mN.m2"
    M_N_KM = "mN/km"
    M_N_M = "mN/m"
    MOHM = "mohm"
    MOHM_1 = "Mohm"
    MOL = "mol"
    MOL_M2_MOL_S = "mol.m2/(mol.s)"
    MOL_S_M2 = "mol/(s.m2)"
    MOL_M2 = "mol/m2"
    MOL_M3 = "mol/m3"
    MOL_MOL = "mol/mol"
    MOL_S = "mol/s"
    M_P = "mP"
    MP_1 = "MP"
    MPA_1 = "MPa"
    M_PA_2 = "mPa"
    M_PA_S = "mPa.s"
    MPA_S_M = "MPa.s/m"
    MPA_H = "MPa/h"
    MPA_M = "MPa/m"
    MPSI = "Mpsi"
    MRAD = "mrad"
    MRAD_1 = "Mrad"
    MRD = "Mrd"
    MRD_1 = "mrd"
    MREM = "mrem"
    MREM_H = "mrem/h"
    MS_1 = "MS"
    M_S_3 = "mS"
    MS_4 = "ms"
    MS_CM = "ms/cm"
    M_S_CM_1 = "mS/cm"
    MS_FT = "ms/ft"
    MS_IN = "ms/in"
    M_S_M = "mS/m"
    MS_M_1 = "ms/m"
    MS_S = "ms/s"
    M_SV = "mSv"
    M_SV_H = "mSv/h"
    M_T = "mT"
    M_T_DM = "mT/dm"
    MV = "MV"
    M_V_1 = "mV"
    M_V_FT = "mV/ft"
    M_V_M = "mV/m"
    MW = "MW"
    M_W_1 = "mW"
    MW_H = "MW.h"
    MW_H_KG = "MW.h/kg"
    MW_H_M3 = "MW.h/m3"
    M_W_M2 = "mW/m2"
    M_WB = "mWb"
    MWB_1 = "MWb"
    N = "N"
    N_M = "N.m"
    N_M_M = "N.m/m"
    N_M2 = "N.m2"
    N_S_M2 = "N.s/m2"
    N_M_1 = "N/m"
    N_M2_1 = "N/m2"
    N_M3 = "N/m3"
    N_MM2 = "N/mm2"
    N_N = "N/N"
    N_A = "nA"
    NA_1 = "na"
    N_API = "nAPI"
    N_C = "nC"
    NCAL_TH = "ncal[th]"
    N_CI = "nCi"
    N_EUC = "nEuc"
    NE_V = "neV"
    N_F = "nF"
    NG = "ng"
    NG_G = "ng/g"
    NG_MG = "ng/mg"
    NGAUSS = "ngauss"
    N_GY = "nGy"
    N_H = "nH"
    N_HZ = "nHz"
    N_J = "nJ"
    NM_4 = "nm"
    NM_S = "nm/s"
    N_N_1 = "nN"
    NOHM = "nohm"
    NOHM_MIL2_FT = "nohm.mil2/ft"
    NOHM_MM2_M = "nohm.mm2/m"
    N_P = "nP"
    N_PA = "nPa"
    NRD = "nrd"
    N_S = "nS"
    NS_1 = "ns"
    NS_FT = "ns/ft"
    NS_M = "ns/m"
    N_T = "nT"
    N_V = "nV"
    N_W = "nW"
    N_WB = "nWb"
    O = "O"
    OE = "Oe"
    OHM = "ohm"
    OHM_CM = "ohm.cm"
    OHM_M = "ohm.m"
    OHM_M2_M = "ohm.m2/m"
    OHM_M_1 = "ohm/m"
    OZF = "ozf"
    OZM = "ozm"
    OZM_TROY = "ozm[troy]"
    P = "P"
    P_A = "pA"
    PA_1 = "Pa"
    PA_S = "Pa.s"
    PA_S_M3_KG = "Pa.s.m3/kg"
    PA_S_M3 = "Pa.s/m3"
    PA_S2_M3 = "Pa.s2/m3"
    PA_H = "Pa/h"
    PA_M = "Pa/m"
    PA_M3 = "Pa/m3"
    PA_S_1 = "Pa/s"
    PA2 = "Pa2"
    PA2_PA_S = "Pa2/(Pa.s)"
    P_C = "pC"
    PCAL_TH = "pcal[th]"
    P_CI = "pCi"
    P_CI_G = "pCi/g"
    PDL = "pdl"
    PDL_CM2 = "pdl.cm2"
    PDL_FT = "pdl.ft"
    PDL_CM = "pdl/cm"
    P_EUC = "pEuc"
    PE_V = "peV"
    P_F = "pF"
    PG = "pg"
    PGAUSS = "pgauss"
    P_GY = "pGy"
    P_HZ = "pHz"
    P_J = "pJ"
    PM = "pm"
    P_N = "pN"
    POHM = "pohm"
    P_P = "pP"
    P_PA = "pPa"
    PPK = "ppk"
    PPM = "ppm"
    PPM_MASS = "ppm[mass]"
    PPM_VOL = "ppm[vol]"
    PPM_VOL_DEG_C = "ppm[vol]/degC"
    PPM_VOL_DEG_F = "ppm[vol]/degF"
    PRD = "prd"
    PS = "ps"
    P_S_1 = "pS"
    PSI = "psi"
    PSI_D_BBL = "psi.d/bbl"
    PSI_S = "psi.s"
    PSI_FT = "psi/ft"
    PSI_H = "psi/h"
    PSI_M = "psi/m"
    PSI_MIN = "psi/min"
    PSI2 = "psi2"
    PSI2_D_C_P_FT3 = "psi2.d/(cP.ft3)"
    PSI2_C_P = "psi2/cP"
    P_T = "pT"
    PT_UK = "pt[UK]"
    PT_UK_HP_H = "pt[UK]/(hp.h)"
    PT_US = "pt[US]"
    P_V = "pV"
    P_W = "pW"
    P_WB = "pWb"
    QT_UK = "qt[UK]"
    QT_US = "qt[US]"
    QUAD = "quad"
    QUAD_A = "quad/a"
    RAD = "rad"
    RAD_FT = "rad/ft"
    RAD_FT3 = "rad/ft3"
    RAD_M = "rad/m"
    RAD_M3 = "rad/m3"
    RAD_S = "rad/s"
    RAD_S2 = "rad/s2"
    RD = "rd"
    REM = "rem"
    REM_H = "rem/h"
    REV = "rev"
    REV_FT = "rev/ft"
    REV_M = "rev/m"
    REV_S = "rev/s"
    ROD_US = "rod[US]"
    RPM = "rpm"
    RPM_S = "rpm/s"
    S = "S"
    S_1 = "s"
    S_CM = "s/cm"
    S_FT = "s/ft"
    S_FT3 = "s/ft3"
    S_IN = "s/in"
    S_KG = "s/kg"
    S_L = "s/L"
    S_M = "S/m"
    S_M_1 = "s/m"
    S_M3 = "s/m3"
    S_QT_UK = "s/qt[UK]"
    S_QT_US = "s/qt[US]"
    S_S = "s/s"
    SACK_94LBM = "sack[94lbm]"
    SECA = "seca"
    SECTION = "section"
    SR = "sr"
    ST = "St"
    SV = "Sv"
    SV_H = "Sv/h"
    SV_S = "Sv/s"
    T = "t"
    T_1 = "T"
    T_A = "t/a"
    T_D = "t/d"
    T_H = "t/h"
    T_M = "T/m"
    T_M3 = "t/m3"
    T_MIN = "t/min"
    TA_1 = "TA"
    TA_T = "Ta[t]"
    TBQ = "TBq"
    TC = "TC"
    TCAL_TH = "Tcal[th]"
    TD_API = "TD[API]"
    TD_API_M = "TD[API].m"
    TD_API_PA_S = "TD[API]/(Pa.s)"
    TEUC = "TEuc"
    TE_V = "TeV"
    TF = "TF"
    TG = "Tg"
    TGAUSS = "Tgauss"
    TGY = "TGy"
    TH_1 = "TH"
    THERM_EC = "therm[EC]"
    THERM_UK = "therm[UK]"
    THERM_US = "therm[US]"
    THZ = "THz"
    TJ = "TJ"
    TJ_A = "TJ/a"
    TM_1 = "Tm"
    TN = "TN"
    TOHM = "Tohm"
    TON_UK = "ton[UK]"
    TON_UK_A = "ton[UK]/a"
    TON_UK_D = "ton[UK]/d"
    TON_UK_H = "ton[UK]/h"
    TON_UK_MIN = "ton[UK]/min"
    TON_US = "ton[US]"
    TON_US_A = "ton[US]/a"
    TON_US_D = "ton[US]/d"
    TON_US_FT2 = "ton[US]/ft2"
    TON_US_H = "ton[US]/h"
    TON_US_MIN = "ton[US]/min"
    TONF_UK = "tonf[UK]"
    TONF_UK_FT2 = "tonf[UK].ft2"
    TONF_UK_FT = "tonf[UK]/ft"
    TONF_UK_FT2_1 = "tonf[UK]/ft2"
    TONF_US = "tonf[US]"
    TONF_US_FT = "tonf[US].ft"
    TONF_US_FT2 = "tonf[US].ft2"
    TONF_US_MI = "tonf[US].mi"
    TONF_US_MI_BBL = "tonf[US].mi/bbl"
    TONF_US_MI_FT = "tonf[US].mi/ft"
    TONF_US_FT_1 = "tonf[US]/ft"
    TONF_US_FT2_1 = "tonf[US]/ft2"
    TONF_US_IN2 = "tonf[US]/in2"
    TON_REFRIG = "tonRefrig"
    TORR = "torr"
    TP = "TP"
    TPA = "TPa"
    TRD = "Trd"
    TS = "TS"
    TT = "TT"
    TV = "TV"
    TW = "TW"
    TW_H = "TW.h"
    TWB = "TWb"
    U_A = "uA"
    U_A_CM2 = "uA/cm2"
    U_A_IN2 = "uA/in2"
    UBAR = "ubar"
    U_C = "uC"
    UCAL_TH = "ucal[th]"
    UCAL_TH_S_CM2 = "ucal[th]/(s.cm2)"
    UCAL_TH_S = "ucal[th]/s"
    U_CI = "uCi"
    U_EUC = "uEuc"
    UE_V = "ueV"
    U_F = "uF"
    U_F_M = "uF/m"
    UG = "ug"
    UG_CM3 = "ug/cm3"
    UG_G = "ug/g"
    UG_MG = "ug/mg"
    UGAUSS = "ugauss"
    U_GY = "uGy"
    U_H = "uH"
    U_H_M = "uH/m"
    U_HZ = "uHz"
    U_J = "uJ"
    UM = "um"
    UM_S = "um/s"
    UM2 = "um2"
    UM2_M = "um2.m"
    UM_HG_0DEG_C = "umHg[0degC]"
    UMOL = "umol"
    U_N = "uN"
    UOHM = "uohm"
    UOHM_FT = "uohm/ft"
    UOHM_M = "uohm/m"
    U_P = "uP"
    U_PA = "uPa"
    UPSI = "upsi"
    URAD = "urad"
    URD = "urd"
    US = "us"
    U_S_1 = "uS"
    US_FT = "us/ft"
    US_IN = "us/in"
    US_M = "us/m"
    U_T = "uT"
    U_V = "uV"
    U_V_FT = "uV/ft"
    U_V_M = "uV/m"
    U_W = "uW"
    U_W_M3 = "uW/m3"
    U_WB = "uWb"
    V = "V"
    V_B = "V/B"
    V_D_B = "V/dB"
    V_M = "V/m"
    W = "W"
    W_M2_K_J_K = "W.m2.K/(J.K)"
    W_M_K = "W/(m.K)"
    W_M2_K = "W/(m2.K)"
    W_M2_SR = "W/(m2.sr)"
    W_M3_K = "W/(m3.K)"
    W_CM2 = "W/cm2"
    W_K = "W/K"
    W_K_W = "W/kW"
    W_M2 = "W/m2"
    W_M3 = "W/m3"
    W_MM2 = "W/mm2"
    W_SR = "W/sr"
    W_W = "W/W"
    WB = "Wb"
    WB_M = "Wb.m"
    WB_M_1 = "Wb/m"
    WB_MM = "Wb/mm"
    WK_1 = "wk"
    YD = "yd"
    YD_BN_A = "yd[BnA]"
    YD_BN_B = "yd[BnB]"
    YD_CLA = "yd[Cla]"
    YD_IND = "yd[Ind]"
    YD_IND37 = "yd[Ind37]"
    YD_IND62 = "yd[Ind62]"
    YD_IND75 = "yd[Ind75]"
    YD_SE = "yd[Se]"
    YD_SE_T = "yd[SeT]"
    YD_US = "yd[US]"
    YD2 = "yd2"
    YD3 = "yd3"


class SequenceStratigraphySurface(Enum):
    """
    The enumerated attributes of a horizon.
    """

    FLOODING = "flooding"
    RAVINEMENT = "ravinement"
    MAXIMUM_FLOODING = "maximum flooding"
    TRANSGRESSIVE = "transgressive"


class StreamlineFlux(Enum):
    """
    Enumeration of the usual streamline fluxes.

    Properties
    ----------
    OIL
        Oil Phase flux
    GAS
        Gas Phase flux
    WATER
        Water Phase flux
    TOTAL
        Sum of (Water + Oil + Gas) Phase fluxes
    OTHER
        Used to indicate that another flux is being traced. BUSINESS RULE:
        OtherFlux should appear if this value is specified.
    """

    OIL = "oil"
    GAS = "gas"
    WATER = "water"
    TOTAL = "total"
    OTHER = "other"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StringLookup:
    """
    Defines an element inside a string-to-integer lookup table.

    Parameters
    ----------
    key
        The corresponding integer value. This value is used in HDF5 instead
        of the string value. The value of null integer value must be
        reserved for NULL. The size of this value is constrained by the size
        of the format used in HDF5,
    value
        A string value. Output from the lookup table.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    key: int = field(
        metadata={
            "name": "Key",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    value: str = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


class SubnodeNodeObject(Enum):
    """SubnodeNodeObject is used to specify the node object that supports the
    subnodes.

    This determines the number of nodes per subnode and the continuity
    of the associated geometry or property. For instance, for hexahedral
    cells, cell indicates a fixed value of 8, while for an unstructured
    column layer grid, cell indicates that this count varies from column
    to column.

    Properties
    ----------
    CELL
        If geometry or properties are discontinuous from cell to cell (i.e.,
        their spatial support is cell), then attach them to cell subnodes.
        BUSINESS RULE: If this object kind is selected, then an ordered list
        of nodes per cell must be specified or otherwise known.
    FACE
        If geometry or properties are continuous between cells that share
        the same face (i.e., their spatial support is the face), then attach
        them to face subnodes. BUSINESS RULE: If this object kind is
        selected, then an ordered list of nodes per face must be specified
        or otherwise known.
    EDGE
        If geometry and properties are continuous between cells that share
        the same edge of a face (i.e. their spatial support is the edge),
        then attach them to edge subnodes. BUSINESS RULE: If this object
        kind is selected, then an ordered list of nodes per edge must be
        specified or otherwise known.
    """

    CELL = "cell"
    FACE = "face"
    EDGE = "edge"


class SurfaceRole(Enum):
    """
    Indicates the various roles that a surface topology can have.

    Properties
    ----------
    MAP
        Representation support for properties.
    PICK
        Representation support for 3D points picked in 2D or 3D.
    """

    MAP = "map"
    PICK = "pick"


class TectonicBoundaryKind(Enum):
    """
    Enumeration of the types of tectonic boundaries.

    Properties
    ----------
    FAULT
        Fracture with displacement
    FRACTURE
        Fracture
    """

    FAULT = "fault"
    FRACTURE = "fracture"


class ThrowKind(Enum):
    """
    Enumerations that characterize the throw of the fault interpretation.
    """

    REVERSE = "reverse"
    NORMAL = "normal"
    THRUST = "thrust"
    STRIKE_AND_SLIP = "strike and slip"
    SCISSOR = "scissor"
    VARIABLE = "variable"


class TimeSetKind(Enum):
    """
    Indicates that the collection of properties shares this time relationship, if
    any.

    Properties
    ----------
    SINGLE_TIME
        Indicates that the collection contains only property values
        associated with a single time index, i.e., time identity can be
        ascertained from the time index itself, without knowledge of the
        time.
    EQUIVALENT_TIMES
        Indicates that the collection of properties is at equivalent times,
        e.g., a 4D seismic data set and a reservoir simulation model at
        comparable times. For a more specific relationship, select single
        time.
    NOT_A_TIME_SET
        Indicates that the property collection is not related by time.
    """

    SINGLE_TIME = "single time"
    EQUIVALENT_TIMES = "equivalent times"
    NOT_A_TIME_SET = "not a time set"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Timestamp:
    """
    Time.

    Parameters
    ----------
    date_time
        A date which can be represented according to the W3CDTF format.
    year_offset
        Indicates that the dateTime attribute must be translated according
        to this value.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    date_time: XmlDateTime = field(
        metadata={
            "name": "DateTime",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    year_offset: Optional[int] = field(
        default=None,
        metadata={
            "name": "YearOffset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractObject_Type:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"

    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    uuid: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Boolean:
    class Meta:
        namespace = "http://www.isotc211.org/2005/gco"

    value: bool = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CharacterString:
    class Meta:
        namespace = "http://www.isotc211.org/2005/gco"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CodeListValue_Type:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"


    code_list: str = field(
        metadata={
            "name": "codeList",
            "type": "Attribute",
            "required": True,
        }
    )
    code_list_value: str = field(
        metadata={
            "name": "codeListValue",
            "type": "Attribute",
            "required": True,
        }
    )
    code_space: Optional[str] = field(
        default=None,
        metadata={
            "name": "codeSpace",
            "type": "Attribute",
        },
    )
    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Date:
    class Meta:
        nillable = True
        namespace = "http://www.isotc211.org/2005/gco"

    value: Optional[Union[XmlDate, XmlPeriod]] = field(
        metadata={
            "nillable": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DateTime:
    class Meta:
        namespace = "http://www.isotc211.org/2005/gco"

    value: XmlDateTime = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Real:
    class Meta:
        namespace = "http://www.isotc211.org/2005/gco"

    value: float = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class URL:
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


class AggregationType(Enum):
    SET = "set"
    BAG = "bag"
    SEQUENCE = "sequence"
    ARRAY = "array"
    RECORD = "record"
    TABLE = "table"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CodeType:
    """Gml:CodeType is a generalized type to be used for a term, keyword or name.

    It adds a XML attribute codeSpace to a term, where the value of the
    codeSpace attribute (if present) shall indicate a dictionary,
    thesaurus, classification scheme, authority, or pattern for the
    term.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    code_space: Optional[str] = field(
        default=None,
        metadata={
            "name": "codeSpace",
            "type": "Attribute",
        },
    )


class NilReasonEnumeration_value(Enum):
    INAPPLICABLE = "inapplicable"
    MISSING = "missing"
    TEMPLATE = "template"
    UNKNOWN = "unknown"
    WITHHELD = "withheld"


class RelatedTimeType_relativePosition(Enum):
    BEFORE = "Before"
    AFTER = "After"
    BEGINS = "Begins"
    ENDS = "Ends"
    DURING = "During"
    EQUALS = "Equals"
    CONTAINS = "Contains"
    OVERLAPS = "Overlaps"
    MEETS = "Meets"
    OVERLAPPED_BY = "OverlappedBy"
    MET_BY = "MetBy"
    BEGUN_BY = "BegunBy"
    ENDED_BY = "EndedBy"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SecondDefiningParameter_1:
    class Meta:
        name = "SecondDefiningParameter"
        namespace = "http://www.opengis.net/gml/3.2"

    inverse_flattening: Optional[float] = field(
        default=None,
        metadata={
            "name": "inverseFlattening",
            "type": "Element",
        },
    )
    semi_minor_axis: Optional[float] = field(
        default=None,
        metadata={
            "name": "semiMinorAxis",
            "type": "Element",
        },
    )
    is_sphere: bool = field(
        default=True,
        metadata={
            "name": "isSphere",
            "type": "Element",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class greenwichLongitude:
    """Gml:greenwichLongitude is the longitude of the prime meridian measured from
    the Greenwich meridian, positive eastward.

    If the value of the prime meridian "name" is "Greenwich" then the
    value of greenwichLongitude shall be 0 degrees.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: float = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class maximumValue:
    """The gml:minimumValue and gml:maximumValue properties allow the specification
    of minimum and maximum value normally allowed for this axis, in the unit of
    measure for the axis.

    For a continuous angular axis such as longitude, the values wrap-
    around at this value. Also, values beyond this minimum/maximum can
    be used for specified purposes, such as in a bounding box. A value
    of minus infinity shall be allowed for the gml:minimumValue element,
    a value of plus infiniy for the gml:maximumValue element. If these
    elements are omitted, the value is unspecified.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: float = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class minimumValue:
    """The gml:minimumValue and gml:maximumValue properties allow the specification
    of minimum and maximum value normally allowed for this axis, in the unit of
    measure for the axis.

    For a continuous angular axis such as longitude, the values wrap-
    around at this value. Also, values beyond this minimum/maximum can
    be used for specified purposes, such as in a bounding box. A value
    of minus infinity shall be allowed for the gml:minimumValue element,
    a value of plus infiniy for the gml:maximumValue element. If these
    elements are omitted, the value is unspecified.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: float = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class operationVersion:
    """Gml:operationVersion is the version of the coordinate transformation (i.e.,
    instantiation due to the stochastic nature of the parameters).

    Mandatory when describing a transformation, and should not be
    supplied for a conversion.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class realizationEpoch:
    """Gml:realizationEpoch is the time after which this datum definition is valid.

    See ISO 19111 Table 32 for details.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: XmlDate = field(
        metadata={
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class remarks:
    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class scope:
    """The gml:scope property provides a description of the usage, or limitations
    of usage, for which this CRS-related object is valid.

    If unknown, enter "not known".
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class semiMajorAxis:
    """Gml:semiMajorAxis specifies the length of the semi-major axis of the
    ellipsoid, with its units.

    Uses the MeasureType with the restriction that the unit of measure
    referenced by uom must be suitable for a length, such as metres or
    feet.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    value: float = field(
        metadata={
            "required": True,
        }
    )


class actuate_value(Enum):
    ON_LOAD = "onLoad"
    ON_REQUEST = "onRequest"
    OTHER = "other"
    NONE = "none"


class show_value(Enum):
    NEW = "new"
    REPLACE = "replace"
    EMBED = "embed"
    OTHER = "other"
    NONE = "none"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class APIGammaRayMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: APIGammaRayUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class APIGravityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: APIGravityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class APINeutronMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: APINeutronUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbsorbedDoseMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AbsorbedDoseUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractObject:
    """The intended abstract supertype of all schema roots that may be a member of
    a substitution group (whether contextual or data).

    The type of root global elements should be extended from this type
    and the root global element should be declared to be a member of one
    of the above substitution groups.

    Parameters
    ----------
    citation
    aliases
    custom_data
    schema_version
        The specific version of a schema from which this object is derived.
        This string should be exactly equivalent to the version attribute of
        the root element of the associated XSD schema file. In the UML model
        is the same as the version tagged value of the
        &lt;&lt;XSDschema&gt;&gt; package.
    uuid
    object_version
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    schema_version: str = field(
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
            "required": True,
        }
    )
    uuid: str = field(
        metadata={
            "type": "Attribute",
            "required": True,
            "pattern": r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
        }
    )
    aliases: List[ObjectAlias] = field(
        default_factory=list,
        metadata={
            "name": "Aliases",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
        },
    )
    object_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "objectVersion",
            "type": "Attribute",
            "min_length": 1,
            "max_length": 64,
            "white_space": "collapse",
        },
    )
    citation: Optional[Citation] = field(
        default=None,
        metadata={
            "name": "Citation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
        },
    )
    custom_data: Optional[CustomData] = field(
        default=None,
        metadata={
            "name": "CustomData",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ActivityOfRadioactivityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ActivityOfRadioactivityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AmountOfSubstanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AmountOfSubstanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AmountOfSubstancePerAmountOfSubstanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AmountOfSubstancePerAmountOfSubstanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AmountOfSubstancePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AmountOfSubstancePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AmountOfSubstancePerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AmountOfSubstancePerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AmountOfSubstancePerTimePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AmountOfSubstancePerTimePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AmountOfSubstancePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AmountOfSubstancePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AnglePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AnglePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AnglePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AnglePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AngularAccelerationMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AngularAccelerationUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AngularVelocityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AngularVelocityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AreaPerAmountOfSubstanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AreaPerAmountOfSubstanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AreaPerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AreaPerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AreaPerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AreaPerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AreaPerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AreaPerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AreaPerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AreaPerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AttenuationPerFrequencyIntervalMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: AttenuationPerFrequencyIntervalUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CapacitanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: CapacitanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DataTransferSpeedMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DataTransferSpeedUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DiffusionCoefficientMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DiffusionCoefficientUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DigitalStorageMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DigitalStorageUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DimensionlessMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DimensionlessUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DipoleMomentMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DipoleMomentUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DoseEquivalentMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DoseEquivalentUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DynamicViscosityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: DynamicViscosityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricChargeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricChargeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricChargePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricChargePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricChargePerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricChargePerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricChargePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricChargePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricConductanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricConductanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricConductivityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricConductivityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricCurrentDensityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricCurrentDensityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricCurrentMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricCurrentUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricFieldStrengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricFieldStrengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricPotentialDifferenceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricPotentialDifferenceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricResistanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricResistanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricResistancePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricResistancePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectricalResistivityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectricalResistivityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElectromagneticMomentMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ElectromagneticMomentUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyLengthPerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyLengthPerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyLengthPerTimeAreaTemperatureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyLengthPerTimeAreaTemperatureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyPerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyPerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyPerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyPerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyPerMassPerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyPerMassPerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EnergyPerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: EnergyPerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ForceAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ForceAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ForceLengthPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ForceLengthPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ForceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ForceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ForcePerForceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ForcePerForceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ForcePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ForcePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ForcePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ForcePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FrequencyIntervalMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: FrequencyIntervalUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FrequencyMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: FrequencyUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Hdf5Dataset:
    """
    Parameters
    ----------
    path_in_hdf_file
        The path of the referenced dataset in the HDF file. The separator
        between groups and final dataset is a slash '/'
    hdf_proxy
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    path_in_hdf_file: str = field(
        metadata={
            "name": "PathInHdfFile",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    hdf_proxy: DataObjectReference = field(
        metadata={
            "name": "HdfProxy",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class HeatCapacityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: HeatCapacityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class HeatFlowRateMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: HeatFlowRateUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class HeatTransferCoefficientMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: HeatTransferCoefficientUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IlluminanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: IlluminanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class InductanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: InductanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IsothermalCompressibilityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: IsothermalCompressibilityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class KinematicViscosityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: KinematicViscosityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthPerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthPerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthPerPressureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthPerPressureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthPerTemperatureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthPerTemperatureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthPerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthPerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LengthPerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LengthPerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LightExposureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LightExposureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LinearAccelerationMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LinearAccelerationUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LinearThermalExpansionMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LinearThermalExpansionUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LogarithmicPowerRatioMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LogarithmicPowerRatioUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LogarithmicPowerRatioPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LogarithmicPowerRatioPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LuminanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LuminanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LuminousEfficacyMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LuminousEfficacyUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LuminousFluxMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LuminousFluxUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LuminousIntensityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: LuminousIntensityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticDipoleMomentMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticDipoleMomentUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticFieldStrengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticFieldStrengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticFluxDensityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticFluxDensityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticFluxDensityPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticFluxDensityPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticFluxMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticFluxUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticPermeabilityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticPermeabilityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MagneticVectorPotentialMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MagneticVectorPotentialUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerEnergyMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerEnergyUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerTimePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerTimePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerTimePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerTimePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MassPerVolumePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MassPerVolumePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MobilityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MobilityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MolarEnergyMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MolarEnergyUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MolarHeatCapacityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MolarHeatCapacityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MolarVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MolarVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MolecularWeightMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MolecularWeightUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MomentOfForceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MomentOfForceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MomentOfInertiaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MomentOfInertiaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MomentumMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: MomentumUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class NormalizedPowerMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: NormalizedPowerUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PermeabilityLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PermeabilityLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PermeabilityRockMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PermeabilityRockUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PermittivityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PermittivityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PlaneAngleMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PlaneAngleUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PotentialDifferencePerPowerDropMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PotentialDifferencePerPowerDropUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PowerMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PowerUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PowerPerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PowerPerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PowerPerPowerMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PowerPerPowerUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PowerPerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PowerPerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PressureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PressureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PressurePerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PressurePerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PressurePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PressurePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PressureSquaredMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PressureSquaredUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PressureSquaredPerForceTimePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PressureSquaredPerForceTimePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PressureTimePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: PressureTimePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ProjectedCrsEpsgCode(AbstractProjectedCrs):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    epsg_code: int = field(
        metadata={
            "name": "EpsgCode",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ProjectedUnknownCrs(AbstractProjectedCrs):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    unknown: str = field(
        metadata={
            "name": "Unknown",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "min_length": 1,
            "max_length": 256,
            "white_space": "collapse",
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class QuantityOfLightMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: QuantityOfLightUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RadianceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: RadianceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RadiantIntensityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: RadiantIntensityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalElectricPotentialDifferenceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalElectricPotentialDifferenceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalForceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalForceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalMassTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalMassTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalPressureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalPressureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReciprocalVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReciprocalVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReluctanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ReluctanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SecondMomentOfAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: SecondMomentOfAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SignalingEventPerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: SignalingEventPerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SolidAngleMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: SolidAngleUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SpecificHeatCapacityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: SpecificHeatCapacityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TemperatureIntervalMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TemperatureIntervalUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TemperatureIntervalPerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TemperatureIntervalPerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TemperatureIntervalPerPressureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TemperatureIntervalPerPressureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TemperatureIntervalPerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TemperatureIntervalPerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThermalConductanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ThermalConductanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThermalConductivityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ThermalConductivityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThermalDiffusivityMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ThermalDiffusivityUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThermalInsulanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ThermalInsulanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThermalResistanceMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ThermalResistanceUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThermodynamicTemperatureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: ThermodynamicTemperatureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TimePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimePerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TimePerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimePerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TimePerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: TimePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCrsEpsgCode(AbstractVerticalCrs):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    epsg_code: int = field(
        metadata={
            "name": "EpsgCode",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalUnknownCrs(AbstractVerticalCrs):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    unknown: str = field(
        metadata={
            "name": "Unknown",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
            "min_length": 1,
            "max_length": 256,
            "white_space": "collapse",
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumeFlowRatePerVolumeFlowRateMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumeFlowRatePerVolumeFlowRateUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerMassMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerMassUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerPressureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerPressureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerRotationMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerRotationUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimeLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimeLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimePerAreaMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimePerAreaUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimePerLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimePerLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimePerPressureLengthMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimePerPressureLengthUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimePerPressureMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimePerPressureUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimePerTimeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimePerTimeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerTimePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerTimePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumePerVolumeMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumePerVolumeUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumetricHeatTransferCoefficientMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumetricHeatTransferCoefficientUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumetricThermalExpansionMeasure:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    value: float = field(
        metadata={
            "required": True,
        }
    )
    uom: VolumetricThermalExpansionUom = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractActivityParameter:
    """
    General parameter value used in one instance of activity.

    Parameters
    ----------
    title
        Name of the parameter, used to identify it in the activity. Must
        have an equivalent in the activity descriptor parameters.
    index
        When parameter is an array, used to indicate the index in the array
    selection
        Textual description about how this parameter was selected.
    key
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    title: str = field(
        metadata={
            "name": "Title",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    selection: Optional[str] = field(
        default=None,
        metadata={
            "name": "Selection",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    key: List[AbstractParameterKey] = field(
        default_factory=list,
        metadata={
            "name": "Key",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractBooleanArray(AbstractValueArray):
    """Generic representation of an array of Boolean values.

    Each derived element provides a specialized implementation to allow
    specific optimization of the representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractContactInterpretationPart:
    """The parent class of an atomic, linear, or surface geologic contact
    description.

    When the contact is between two surface representations (e.g.,
    fault/fault, horizon/fault, horizon/horizon), then the contact is a
    line. When the contact is between two volume representations
    (stratigraphic unit/stratigraphic unit), then the contact is a
    surface. A contact interpretation can be associated with other
    contact interpretations in an organization interpretation. To define
    a contact representation, you must first define a contact
    interpretation.

    Parameters
    ----------
    contact_relationship
    index
        contact index
    part_of
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    contact_relationship: ContactRelationship = field(
        metadata={
            "name": "ContactRelationship",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    part_of: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "PartOf",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractDoubleArray(AbstractValueArray):
    """Generic representation of an array of double values.

    Each derived element provides specialized implementation to allow
    specific optimization of the representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractIntegerArray(AbstractValueArray):
    """Generic representation of an array of integer values.

    Each derived element provides specialized implementation to allow
    specific optimization of the representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractSeismicCoordinates:
    """Parent class is used to associate horizon and fault representations to
    seismic 2D and seismic 3D technical features.

    It stores a 1-to-1 mapping between geometry coordinates (usually X,
    Y, Z) and trace or inter-trace positions on a seismic survey.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    seismic_support: DataObjectReference = field(
        metadata={
            "name": "SeismicSupport",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AdditionalGridPoints:
    """
    Geometry given by means of points attached to additional elements of a grid.

    Parameters
    ----------
    representation_patch_index
        Used to remove ambiguity in geometry attachment, if the attachment
        element is not sufficient. Usually required for subnodes and for the
        general purpose grid, but not otherwise.
    attachment
    points
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    attachment: GridGeometryAttachment = field(
        metadata={
            "name": "Attachment",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    points: AbstractPoint3dArray = field(
        metadata={
            "name": "Points",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation_patch_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "RepresentationPatchIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ChronostratigraphicRank:
    """The chronostratigraphic ranking of “well known” stratigraphic unit features
    in the global chronostratigraphic column.

    The ranks are organized from container to contained, e.g., (eon=1), (era=2), (period=3)
    The units are ranked by using age as ordering criteria, from oldest to youngest.
    These stratigraphic units have no associated interpretations or representations.
    BUSINESS RULE: The name must reference a well-known stratigraphic unit feature (such as "Jurassic"), for example, from the International Commission on Stratigraphy (http://www.stratigraphy.org).

    Parameters
    ----------
    name
        Name of the chrono rank such as "epoch, era, ..."
    contains
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    name: str = field(
        metadata={
            "name": "Name",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
            "min_length": 1,
            "max_length": 64,
            "white_space": "collapse",
        }
    )
    contains: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Contains",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ContactElementReference(DataObjectReference):
    """A reference to either a geologic feature interpretation or a frontier
    feature.

    BUSINESS RULE: The ContentType of the corresponding data-object reference must be a geological feature interpretation or a frontier feature.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    qualifier: Optional[ContactSide] = field(
        default=None,
        metadata={
            "name": "Qualifier",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    secondary_qualifier: Optional[ContactMode] = field(
        default=None,
        metadata={
            "name": "SecondaryQualifier",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ContactRepresentationReference(AbstractContactRepresentationPart):
    """
    Used when the contact already exists as a top level element representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    representation: DataObjectReference = field(
        metadata={
            "name": "Representation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeologicUnitInterpretationIndex:
    """Element that lets you index and order rock feature interpretations.

    For possible ordering criteria, see OrderingCriteria.

    Parameters
    ----------
    index
        An index value associated to an instance of this type
        interpretation, given a specific ordering criteria.
    unit
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    unit: DataObjectReference = field(
        metadata={
            "name": "Unit",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class HorizonInterpretationIndex:
    """Element that lets you index and order horizon interpretations.

    For possible ordering criteria, see OrderingCriteria.

    Parameters
    ----------
    index
        An index value associated to an instance of this type of
        interpretation, given a specific ordering criteria
    stratigraphic_rank
        Number of the stratigraphic rank on which the previous indices have
        been defined.
    horizon
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    horizon: DataObjectReference = field(
        metadata={
            "name": "Horizon",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    stratigraphic_rank: Optional[int] = field(
        default=None,
        metadata={
            "name": "StratigraphicRank",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LocalPropertyKind(AbstractPropertyKind):
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    local_property_kind: DataObjectReference = field(
        metadata={
            "name": "LocalPropertyKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ObjectParameterKey(AbstractParameterKey):
    """
    Parameters
    ----------
    data_object
        Is actually a reference and not a containment relationship.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    data_object: DataObjectReference = field(
        metadata={
            "name": "DataObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Patch1d(Patch):
    """
    A patch with a single 1D index count.

    Parameters
    ----------
    count
        Number of items in the patch.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PatchBoundaries:
    """Defines the boundaries of an indexed patch.

    These boundaries are outer and inner rings.

    Parameters
    ----------
    inner_ring
        A hole inside a representation patch. Inside the ring, the
        representation patch is not defined, outside it is. In case of
        contact, inner ring polyline representations should be typed as an
        erosion line, deposition line, or contact. BUSINESS RULE: Must be a
        polyline reference to a polyline representation, either a single
        polyline representation or a subrepresentation. Must be closed.
    outer_ring
        The extension of a representation patch. Inside the ring, the
        representation patch is defined, outside it is not. BUSINESS RULE:
        Must be a reference to a polyline, either a single polyline
        representation or a subrepresentation. Must be closed.
    referenced_patch
        UUID of the referenced topological patch.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    referenced_patch: int = field(
        metadata={
            "name": "ReferencedPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    inner_ring: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "InnerRing",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    outer_ring: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "OuterRing",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PatchOfPoints:
    """A patch of points.

    In RESQML, a patch is a set or range of one kind of topological
    elements used to define part of a data-object, such as grids or
    structural data-objects.

    Parameters
    ----------
    representation_patch_index
        Optional patch index used to attach properties to a specific patch
        of the indexable elements.
    points
        Geometric points (or vectors) to be attached to the specified
        indexable elements.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    points: AbstractPoint3dArray = field(
        metadata={
            "name": "Points",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation_patch_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "RepresentationPatchIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PatchOfValues:
    """
    A patch of values.

    Parameters
    ----------
    representation_patch_index
        Patch index used to attach properties to a specific patch of the
        indexable elements.
    values
        Values to be attached to the indexable elements.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    values: AbstractValueArray = field(
        metadata={
            "name": "Values",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation_patch_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "RepresentationPatchIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PropertyKindFacet:
    """Qualifiers for property values, which allows users to semantically
    specialize a property without creating a new property kind.

    For the list of enumerations, see Facet.

    Parameters
    ----------
    facet
        Facet of the property kind (see the enumeration)
    value
        Property facet value.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    facet: Facet = field(
        metadata={
            "name": "Facet",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    value: str = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PropertyValuesPatch:
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_uid: int = field(
        metadata={
            "name": "patchUid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    values: AbstractValueArray = field(
        metadata={
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RockFluidUnitInterpretationIndex:
    """
    An element that allows ordering of fluid feature interpretations in a fluid
    organization interpretation.

    Parameters
    ----------
    index
        Index of the fluid feature interpretation.
    rock_fluid_unit
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    rock_fluid_unit: DataObjectReference = field(
        metadata={
            "name": "RockFluidUnit",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StandardPropertyKind(AbstractPropertyKind):
    """A standard property kind is defined in the Energistics catalog.

    It has an unique name.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    kind: ResqmlPropertyKind = field(
        metadata={
            "name": "Kind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StratigraphicUnitInterpretationIndex:
    """Element that lets you index and order stratigraphic unit interpretations.

    For possible ordering criteria, see OrderingCriteria.

    Parameters
    ----------
    index
        An index value associated to an instance of this type of
        interpretation, given a specific ordering criteria.
    unit
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    unit: DataObjectReference = field(
        metadata={
            "name": "Unit",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SubnodePatch(Patch):
    """Each patch of subnodes is defined independently of the others.

    Number of nodes per object is determined by the subnode kind.

    Parameters
    ----------
    subnode_node_object
    node_weights_per_subnode
        Node weights for each subnode. Count of nodes per subnode is known
        for each specific subnode construction. Data order consists of all
        the nodes for each subnode in turn. For example, if uniform and
        stored as a multi-dimensional array, the node index cycles first.
        BUSINESS RULE: Weights must be non-negative. BUSINESS RULE: Length
        of array must be consistent with the sum of nodeCount x subnodeCount
        per object, e.g., for 3 subnodes per edge (uniform), there are 6
        weights.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    subnode_node_object: SubnodeNodeObject = field(
        metadata={
            "name": "SubnodeNodeObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    node_weights_per_subnode: AbstractValueArray = field(
        metadata={
            "name": "NodeWeightsPerSubnode",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ThreePoint3d:
    """
    List of three 3D points.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    point3d: List[Point3d] = field(
        default_factory=list,
        metadata={
            "name": "Point3d",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 3,
            "max_occurs": 3,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeIndex:
    """Index into a time series.

    Used to specify time. (Not to be confused with time step.)

    Parameters
    ----------
    index
        The index of the time in the time series.
    time_series
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    time_series: DataObjectReference = field(
        metadata={
            "name": "TimeSeries",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeInterval:
    """Geological time during which a geological event (e.g., deposition, erosion,
    fracturation, faulting, intrusion) occurred.

    BUSINESS RULE: All rock features that are present in the global chronostratigraphic column feature must have a time interval.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    chrono_bottom: DataObjectReference = field(
        metadata={
            "name": "ChronoBottom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    chrono_top: DataObjectReference = field(
        metadata={
            "name": "ChronoTop",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumeShell:
    """
    The shell or envelope of a structural, stratigraphic, or fluid unit.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    shell_uid: str = field(
        metadata={
            "name": "ShellUid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    macro_faces: List[OrientedMacroFace] = field(
        default_factory=list,
        metadata={
            "name": "MacroFaces",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreTrajectoryParentIntersection:
    """
    For a wellbore trajectory in a multi-lateral well, indicates the MD of the
    kickoff point where the trajectory begins and the corresponding MD of the
    parent trajectory.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    kickoff_md: float = field(
        metadata={
            "name": "KickoffMd",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_md: float = field(
        metadata={
            "name": "ParentMd",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_trajectory: DataObjectReference = field(
        metadata={
            "name": "ParentTrajectory",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WitsmlWellboreReference:
    """
    Reference to the WITSML wellbore that this wellbore feature is based on.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    witsml_well: DataObjectReference = field(
        metadata={
            "name": "WitsmlWell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    witsml_wellbore: DataObjectReference = field(
        metadata={
            "name": "WitsmlWellbore",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Boolean_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"

    boolean: Optional[bool] = field(
        default=None,
        metadata={
            "name": "Boolean",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gco",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DateTime_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"

    date_time: Optional[XmlDateTime] = field(
        default=None,
        metadata={
            "name": "DateTime",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gco",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Date_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"

    date: Optional[Union[XmlDate, XmlPeriod]] = field(
        default=None,
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gco",
            "nillable": True,
        },
    )
    date_time: Optional[XmlDateTime] = field(
        default=None,
        metadata={
            "name": "DateTime",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gco",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Real_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"

    real: Optional[float] = field(
        default=None,
        metadata={
            "name": "Real",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gco",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractDQ_Result_Type(AbstractObject_Type):
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_DateTypeCode(CodeListValue_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_OnLineFunctionCode(CodeListValue_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_PresentationFormCode(CodeListValue_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_RoleCode(CodeListValue_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DQ_EvaluationMethodTypeCode(CodeListValue_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DQ_Result_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_GeographicExtent_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class URL_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    url: Optional[str] = field(
        default=None,
        metadata={
            "name": "URL",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TM_Primitive_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gts"

    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CodeWithAuthorityType(CodeType):
    """
    Gml:CodeWithAuthorityType requires that the codeSpace attribute is provided in
    an instance.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    code_space: str = field(
        default=None,
        metadata={
            "name": "codeSpace",
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeneralConversionPropertyType:
    """
    Gml:GeneralConversionPropertyType is a property type for association roles to a
    general conversion, either referencing or containing the definition of that
    conversion.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ReferenceType:
    """
    Gml:ReferenceType is intended to be used in application schemas directly, if a
    property element shall use a "by-reference only" encoding.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    owns: bool = field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StringOrRefType:
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimePrimitivePropertyType:
    """
    Gml:TimePrimitivePropertyType provides a standard content model for
    associations between an arbitrary member of the substitution group whose head
    is gml:AbstractTimePrimitive and another object.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )
    owns: bool = field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class anchorDefinition(CodeType):
    """Gml:anchorDefinition is a description, possibly including coordinates, of
    the definition used to anchor the datum to the Earth. Also known as the
    "origin", especially for engineering and image datums. The codeSpace attribute
    may be used to reference a source of more detailed on this point or surface, or
    on a set of such descriptions.

    -       For a geodetic datum, this point is also known as the fundamental point, which is traditionally the point where the relationship between geoid and ellipsoid is defined. In some cases, the "fundamental point" may consist of a number of points. In those cases, the parameters defining the geoid/ellipsoid relationship have been averaged for these points, and the averages adopted as the datum definition.
    -       For an engineering datum, the anchor definition may be a physical point, or it may be a point with defined coordinates in another CRS.may
    -       For an image datum, the anchor definition is usually either the centre of the image or the corner of the image.
    -       For a temporal datum, this attribute is not defined. Instead of the anchor definition, a temporal datum carries a separate time origin of type DateTime.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class axisAbbrev(CodeType):
    """Gml:axisAbbrev is the abbreviation used for this coordinate system axis;
    this abbreviation is also used to identify the coordinates in the coordinate
    tuple.

    The codeSpace attribute may reference a source of more information
    on a set of standardized abbreviations, or on this abbreviation.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class coordinateOperationAccuracy:
    """Gml:coordinateOperationAccuracy is an association role to a
    DQ_PositionalAccuracy object as encoded in ISO/TS 19139, either referencing or
    containing the definition of that positional accuracy.

    That object contains an estimate of the impact of this coordinate
    operation on point accuracy. That is, it gives position error
    estimates for the target coordinates of this coordinate operation,
    assuming no errors in the source coordinates.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class name(CodeType):
    """The gml:name property provides a label or identifier for the object,
    commonly a descriptive name.

    An object may have several names, typically assigned by different
    authorities. gml:name uses the gml:CodeType content model.  The
    authority for a name is indicated by the value of its (optional)
    codeSpace attribute.  The name may or may not be unique, as
    determined by the rules of the organization responsible for the
    codeSpace.  In common usage there will be one name per authority, so
    a processing application may select the name from its preferred
    codeSpace.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class secondDefiningParameter_2:
    """Gml:secondDefiningParameter is a property containing the definition of the
    second parameter that defines the shape of an ellipsoid.

    An ellipsoid requires two defining parameters: semi-major axis and inverse flattening or semi-major axis and semi-minor axis. When the reference body is a sphere rather than an ellipsoid, only a single defining parameter is required, namely the radius of the sphere; in that case, the semi-major axis "degenerates" into the radius of the sphere.
    The inverseFlattening element contains the inverse flattening value of the ellipsoid. This value is a scale factor (or ratio). It uses gml:LengthType with the restriction that the unit of measure referenced by the uom attribute must be suitable for a scale factor, such as percent, permil, or parts-per-million.
    The semiMinorAxis element contains the length of the semi-minor axis of the ellipsoid. When the isSphere element is included, the ellipsoid is degenerate and is actually a sphere. The sphere is completely defined by the semi-major axis, which is the radius of the sphere.
    """

    class Meta:
        name = "secondDefiningParameter"
        namespace = "http://www.opengis.net/gml/3.2"

    second_defining_parameter: SecondDefiningParameter_1 = field(
        metadata={
            "name": "SecondDefiningParameter",
            "type": "Element",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractCitedDataObject(AbstractObject):
    """The Mother Class for all Top Level Elements in RESQML.

    Inherits from the commonv2 AbstractDataObject. The purpose of this
    derivation is simply to make the Citation element mandatory.
    Appropriate to use as a base class in any ML where this is desired.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    citation: Citation = field(
        default=None,
        metadata={
            "name": "Citation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractContextualObject(AbstractObject):
    """
    Substitution group for contextual objects.
    """

    class Meta:
        namespace = "http://www.energistics.org/energyml/data/commonv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractDataObject(AbstractObject):
    """
    Substitution group for normative data objects.
    """

    class Meta:
        namespace = "http://www.energistics.org/energyml/data/commonv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGeometry:
    """
    The base class for all geometric values, which is always associated with a
    representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    local_crs: DataObjectReference = field(
        metadata={
            "name": "LocalCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    time_index: Optional[TimeIndex] = field(
        default=None,
        metadata={
            "name": "TimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Activation:
    """Used to activate and deactivate the referencing object at the times
    indicated.

    If the activation object is not present, then the referencing object
    is always active. If the activation object is present, then the
    referencing object is not active until activated.

    Parameters
    ----------
    activation_toggle_indices
        The index in the time series at which the state of the referencing
        object is changed. Toggle will change state from inactive to active,
        or toggle will change state from active to inactive.
    time_series
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    activation_toggle_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ActivationToggleIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    time_series: DataObjectReference = field(
        metadata={
            "name": "TimeSeries",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BinaryContactInterpretationPart(AbstractContactInterpretationPart):
    """The main class for data describing an opinion of the contact between two
    geologic feature interpretations.

    A contact interpretation between two surface geological boundaries
    is usually a line. A contact interpretation between two volumes
    (rock feature interpretation) is usually a surface. This class
    allows you to build a formal sentence—in the pattern of subject-
    verb-direct object—which is used to describe the construction of a
    node, line, or surface contact. It is also possible to attach a
    primary and a secondary qualifier to the subject and to the direct
    object. For example, one contact interpretation can be described by
    a sentence such as: The interpreted fault named F1 interp on its
    hanging wall side splits the interpreted horizon named H1 Interp on
    both its sides. Subject = F1 Interp, with qualifier "hanging wall
    side" Verb = splits Direct Object = H1 Interp, with qualifier "on
    both sides"

    Parameters
    ----------
    direct_object
        Data-object reference (by UUID link) to a geologic feature
        interpretation, which is the direct object of the sentence that
        defines how the contact was constructed.
    verb
    subject
        Data-object reference (by UUID link) to a geologic feature
        interpretation, which is the subject of the sentence that defines
        how the contact was constructed.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    direct_object: ContactElementReference = field(
        default = None,
        metadata={
            "name": "DirectObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    verb: ContactVerb = field(
        default = None,
        metadata={
            "name": "Verb",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    subject: ContactElementReference = field(
        default = None,
        metadata={
            "name": "Subject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BooleanArrayFromDiscretePropertyArray(AbstractBooleanArray):
    """An array of Boolean values that is explicitly defined by indicating which
    indices in the array are either true or false.

    This class is used to represent very sparse true or false data,
    based on a discrete property.

    Parameters
    ----------
    value
        Integer to match for the value to be considered true
    property
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: int = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    property: DataObjectReference = field(
        metadata={
            "name": "Property",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BooleanArrayFromIndexArray(AbstractBooleanArray):
    """An array of Boolean values defined by specifying explicitly which indices in
    the array are either true or false.

    This class is used to represent very sparse true or false data.

    Parameters
    ----------
    count
        Total number of Boolean elements in the array. This number is
        different from the number of indices used to represent the array.
    indices
        Array of integer indices. BUSINESS RULE: Must be non-negative.
    index_is_true
        Indicates whether the specified elements are true or false.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    indices: AbstractIntegerArray = field(
        metadata={
            "name": "Indices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    index_is_true: bool = field(
        metadata={
            "name": "IndexIsTrue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BooleanConstantArray(AbstractBooleanArray):
    """Represents an array of Boolean values where all values are identical.

    This an optimization for which an array of explicit Boolean values
    is not required.

    Parameters
    ----------
    value
        Value inside all the elements of the array.
    count
        Size of the array.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: bool = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BooleanHdf5Array(AbstractBooleanArray):
    """
    Array of boolean values provided explicitly by an HDF5 dataset.

    Parameters
    ----------
    values
        Reference to an HDF5 array of values.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    values: Hdf5Dataset = field(
        metadata={
            "name": "Values",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CellFluidPhaseUnits:
    """
    A mapping from cells to fluid phase unit interpretation to describe the initial
    hydrostatic fluid column.

    Parameters
    ----------
    phase_unit_indices
        Index of the phase unit kind within a given fluid phase organization
        for each cell. Follows the indexing defined by the PhaseUnit
        enumeration. When applied to the wellbore frame representation, the
        indexing is identical to the number of intervals. Use null (-1) if
        no fluid phase is present, e.g., within the seal. BUSINESS RULE:
        Array length is equal to the number of cells in the representation
        (grid, wellbore frame or blocked well).
    fluid_organization
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    phase_unit_indices: AbstractIntegerArray = field(
        metadata={
            "name": "PhaseUnitIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    fluid_organization: DataObjectReference = field(
        metadata={
            "name": "FluidOrganization",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CellStratigraphicUnits:
    """
    A mapping from cell to stratigraphic unit interpretation for a representations
    (grids or blocked wells).

    Parameters
    ----------
    unit_indices
        Index of the stratigraphic unit of a given stratigraphic column for
        each cell. Use null (-1) if no stratigraphic column, e.g., within
        salt BUSINESS RULE: Array length is the number of cells in the grid
        or the blocked well
    stratigraphic_organization
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    unit_indices: AbstractIntegerArray = field(
        metadata={
            "name": "UnitIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    stratigraphic_organization: DataObjectReference = field(
        metadata={
            "name": "StratigraphicOrganization",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ColumnLayerSplitColumnEdges:
    """Column edges are needed to construct the indices for the cell faces for
    column layer grids.

    For split column layer grids, the column edge indices must be
    defined explicitly. Column edges are not required to describe the
    lowest order grid geometry, but may be required for higher order
    geometries or properties.

    Parameters
    ----------
    count
        Number of split column edges in this grid. Must be positive.
    parent_column_edge_indices
        Parent unsplit column edge index for each of the split column edges.
        Used to implicitly define split face indexing.
    column_per_split_column_edge
        Column index for each of the split column edges. Used to implicitly
        define column and cell faces. List-of-lists construction not
        required since each split column edge must be in a single column.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_column_edge_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ParentColumnEdgeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    column_per_split_column_edge: AbstractIntegerArray = field(
        metadata={
            "name": "ColumnPerSplitColumnEdge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ColumnSubnodePatch(SubnodePatch):
    """
    Use this subnode construction if the number of subnodes per object varies from
    column to column, but does not vary from layer to layer.

    Parameters
    ----------
    subnode_count_per_object
        Number of subnodes per object, with a different number in each
        column of the grid.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    subnode_count_per_object: AbstractIntegerArray = field(
        metadata={
            "name": "SubnodeCountPerObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ContactIdentity:
    """Indicates identity between two (or more) contacts.

    For possible types of identities, see IdentityKind.

    Parameters
    ----------
    identity_kind
    list_of_contact_representations
        The contact representations that share common identity as specified
        by their indices
    list_of_identical_nodes
        Indicates which nodes (identified by their common index in all
        contact representations) of the contact representations are
        identical. If this list is not present, then it indicates that all
        nodes in each representation are identical, on an element by element
        level.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    identity_kind: IdentityKind = field(
        metadata={
            "name": "IdentityKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    list_of_contact_representations: AbstractIntegerArray = field(
        metadata={
            "name": "ListOfContactRepresentations",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    list_of_identical_nodes: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "ListOfIdenticalNodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ContactPatch(Patch1d):
    """
    A subset of topological elements of an existing contact representation part
    (sealed or non-sealed contact).

    Parameters
    ----------
    representation_index
        Identifies a representation by its index, in the list of
        representations contained in the organization.
    supporting_representation_nodes
        The ordered list of nodes (identified by their global index) in the
        supporting representation, which constitutes the contact patch.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    representation_index: int = field(
        metadata={
            "name": "RepresentationIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    supporting_representation_nodes: AbstractIntegerArray = field(
        metadata={
            "name": "SupportingRepresentationNodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DataObjectParameter(AbstractActivityParameter):
    """
    Parameter referencing to a top level object.

    Parameters
    ----------
    data_object
        Is actually a reference and not a containment relationship.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    data_object: DataObjectReference = field(
        default = None,
        metadata={
            "name": "DataObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DoubleConstantArray(AbstractDoubleArray):
    """Represents an array of double values where all values are identical.

    This an optimization for which an array of explicit double values is
    not required.

    Parameters
    ----------
    value
        Values inside all the elements of the array.
    count
        Size of the array.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: float = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DoubleHdf5Array(AbstractDoubleArray):
    """An array of double values provided explicitly by an HDF5 dataset.

    By convention, the null value is NaN.

    Parameters
    ----------
    values
        Reference to an HDF5 array of doubles.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    values: Hdf5Dataset = field(
        metadata={
            "name": "Values",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EdgePatch(Patch1d):
    """Describes edges that are not linked to any other edge.

    Because edges do not have indices, a consecutive pair of nodes is
    used to identify each edge. The split edges dataset is a set of
    nodes (2 nodes per edge). Each patch has a set of 2 nodes.

    Parameters
    ----------
    split_edges
        An array of split edges to define patches. It points to an HDF5
        dataset, which must be a 2D array of non-negative integers with
        dimensions 2 x numSplitEdges. integers with dimensions {2,
        numSplitEdges}
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    split_edges: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "SplitEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Edges:
    """Unstructured cell grids require the definition of edges if the subnode
    attachment is of kind edges.

    Use Case: Finite elements, especially for higher order geometry.
    BUSINESS RULE: Edges must be defined for unstructured cell grids if subnode nodes of kind edges are used.

    Parameters
    ----------
    count
        Number of edges. Must be positive.
    nodes_per_edge
        Defines a list of 2 nodes per edge. Count = 2 x EdgeCount
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    nodes_per_edge: AbstractIntegerArray = field(
        metadata={
            "name": "NodesPerEdge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElementIdentity:
    """Indicates the nature of the relationship between 2 or more representations,
    specifically if they are partially or totally identical.

    For possible types of relationships, see IdentityKind. Commonly used
    to identify contacts between representations in model descriptions.
    May also be used to relate the components of a grid (e.g., pillars)
    to those of a structural framework.

    Parameters
    ----------
    element_indices
        Indicates which elements are identical based on their indices in the
        (sub)representation. If not given, then the selected indexable
        elements of each of the selected representations are identical at
        the element by element level. If not given, then all elements are
        specified to be identical. BUSINESS RULE: Number of identical
        elements must equal identicalElementCount for each representation.
    identity_kind
    indexable_element
    representation
    from_time_index
    to_time_index
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    identity_kind: IdentityKind = field(
        metadata={
            "name": "IdentityKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    indexable_element: IndexableElements = field(
        metadata={
            "name": "IndexableElement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation: DataObjectReference = field(
        metadata={
            "name": "Representation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    from_time_index: Optional[TimeIndex] = field(
        default=None,
        metadata={
            "name": "FromTimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    to_time_index: Optional[TimeIndex] = field(
        default=None,
        metadata={
            "name": "ToTimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    element_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "ElementIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ElementIndices:
    """
    Index into the indexable elements selected.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    indexable_element: IndexableElements = field(
        metadata={
            "name": "IndexableElement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    indices: AbstractIntegerArray = field(
        metadata={
            "name": "Indices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FaultThrow:
    """
    Identifies the characteristic of the throw of a fault interpretation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    throw: List[ThrowKind] = field(
        default_factory=list,
        metadata={
            "name": "Throw",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    has_occured_during: Optional[TimeInterval] = field(
        default=None,
        metadata={
            "name": "HasOccuredDuring",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FloatingPointQuantityParameter(AbstractActivityParameter):
    """
    Parameter containing a double value.

    Parameters
    ----------
    value
        Double value
    uom
        Unit of measure associated with the value
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: float = field(
        default=None,
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    uom: ResqmlUom = field(
        default=None,
        metadata={
            "name": "Uom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntegerArrayFromBooleanMaskArray(AbstractIntegerArray):
    """
    One-dimensional array of integer values obtained from the true elements of the
    Boolean mask.

    Parameters
    ----------
    total_index_count
        Total number of integer elements in the array. This number is
        different from the number of Boolean mask values used to represent
        the array.
    mask
        Boolean mask. A true element indicates that the index is included on
        the list of integer values.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    total_index_count: int = field(
        metadata={
            "name": "TotalIndexCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    mask: AbstractBooleanArray = field(
        metadata={
            "name": "Mask",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntegerConstantArray(AbstractIntegerArray):
    """Represents an array of integer values where all values are identical.

    This an optimization for which an array of explicit integer values
    is not required.

    Parameters
    ----------
    value
        Values inside all the elements of the array.
    count
        Size of the array.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: int = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntegerHdf5Array(AbstractIntegerArray):
    """Array of integer values provided explicitly by a HDF5 dataset.

    The null value is explicitly provided. WHERE IS THE NULL VALUE
    SPECIFIED?

    Parameters
    ----------
    null_value
    values
        Reference to an HDF5 array of integers or doubles.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    null_value: int = field(
        metadata={
            "name": "NullValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    values: Hdf5Dataset = field(
        metadata={
            "name": "Values",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntegerQuantityParameter(AbstractActivityParameter):
    """
    Parameter containing an integer value.

    Parameters
    ----------
    value
        Integer value
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: int = field(
        default=None,
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntegerRangeArray(AbstractIntegerArray):
    """Defines an array as a range of integers.

    The range is defined by an initial value and a count defining the
    size of the range.

    Parameters
    ----------
    count
        Size of the array.
    value
        Start value for the range. End value is start+count-1.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    value: int = field(
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntervalGridCells:
    """Specifies the (Grid,Cell) intersection of each Interval of the
    representation, if any.

    The information allows you to locate, on one or several grids, the
    intersection of volume (cells) and surface (faces) elements with a
    wellbore trajectory (existing or planned), streamline trajectories,
    or any polyline set.

    Parameters
    ----------
    cell_count
        The number of non-null entries in the grid indices array.
    grid_indices
        Size of array = IntervalCount. Null values of -1 signify that that
        interval is not within a grid. BUSINESS RULE: The cell count must
        equal the number of non-null entries in this array.
    cell_indices
        The cell index for each interval of a representation. The grid index
        is specified by grid index array, to give the (Grid,Cell) index
        pair. BUSINESS RULE: Array length must equal cell count.
    local_face_pair_per_cell_indices
        For each cell, these are the entry and exit intersection faces of
        the trajectory in the cell. Use null (-1) for missing intersections,
        e.g., when a trajectory originates or terminates within a cell. The
        local face-per-cell index is used because a global face index need
        not have been defined on the grid. BUSINESS RULE: The array
        dimensions must equal 2 x CellCount.
    grids
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_count: int = field(
        metadata={
            "name": "CellCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    grid_indices: AbstractIntegerArray = field(
        metadata={
            "name": "GridIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cell_indices: AbstractIntegerArray = field(
        metadata={
            "name": "CellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    local_face_pair_per_cell_indices: AbstractIntegerArray = field(
        metadata={
            "name": "LocalFacePairPerCellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    grids: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Grids",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntervalStratigraphicUnits:
    """
    A mapping from intervals to stratigraphic units for representations (grids or
    wellbore frames).

    Parameters
    ----------
    unit_indices
        Index of the stratigraphic unit per interval, of a given
        stratigraphic column. Notes: 1.) For grids, intervals = layers + K
        gaps. 2.) If there is no stratigraphic column, e.g., within salt,
        use null (-1) BUSINESS RULE: Array length must equal the number of
        INTERVALS.
    stratigraphic_organization
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    unit_indices: AbstractIntegerArray = field(
        metadata={
            "name": "UnitIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    stratigraphic_organization: DataObjectReference = field(
        metadata={
            "name": "StratigraphicOrganization",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Intervals:
    """Refinement and/or Coarsening per interval.

    If there is a 1:1 correspondence between the parent and child cells,
    then this object is not needed.

    Parameters
    ----------
    interval_count
        The number of intervals in the regrid description. Must be positive.
    parent_count_per_interval
        The number of parent cells in each interval. BUSINESS RULES: 1.) The
        array length must be equal to intervalCount. 2.) For the given
        parentIndex, the total count of parent cells should not extend
        beyond the boundary of the parent grid.
    child_count_per_interval
        The number of child cells in each interval. If the child grid type
        is not commensurate with the parent type, then this attribute is
        ignored by a reader, and its value should be set to null (-1). For
        example, for a parent IJK grid with a child unstructured column
        layer grid, then the child count is non-null for a K regrid, but
        null for an I or J regrid. BUSINESS RULES: 1.) The array length must
        be equal to intervalCount. 2.) If the child grid type is
        commensurate with the parent grid, then the sum of values over all
        intervals must be equal to the corresponding child grid dimension.
    child_cell_weights
        Weights that are proportional to the relative sizes of child cells
        within each interval. The weights need not be normalized.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    interval_count: int = field(
        metadata={
            "name": "IntervalCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_count_per_interval: AbstractIntegerArray = field(
        metadata={
            "name": "ParentCountPerInterval",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    child_count_per_interval: AbstractIntegerArray = field(
        metadata={
            "name": "ChildCountPerInterval",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    child_cell_weights: Optional[AbstractDoubleArray] = field(
        default=None,
        metadata={
            "name": "ChildCellWeights",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class KGaps:
    """Optional object used to indicate that there are global gaps between layers
    in the grid.

    With K gaps, the bottom of one layer need not be continuous with the
    top of the next layer, so the resulting number of intervals is
    greater than the number of layers.

    Parameters
    ----------
    count
        Number of gaps between layers. Must be positive. Number of INTERVALS
        = gapCount + NK.
    gap_after_layer
        Boolean array of length NK-1. TRUE if there is a gap after the
        corresponding layer. NKL = NK + gapCount + 1 BUSINESS RULE: gapCount
        must be consistent with the number of gaps specified by the
        gapAfterLayer array.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: Optional[int] = field(
        default=None,
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    gap_after_layer: Optional[AbstractBooleanArray] = field(
        default=None,
        metadata={
            "name": "GapAfterLayer",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MultipleContactInterpretationPart(AbstractContactInterpretationPart):
    """Describes multiple interface contacts of geologic feature interpretations
    (compared to a binary contact).

    A composition of several contact interpretations.

    Parameters
    ----------
    with_value
        Indicates a list of binary contacts (by their UUIDs) that
        participate in this multiple contact.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    with_value: List[int] = field(
        default_factory=list,
        metadata={
            "name": "With",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class OverlapVolume:
    """Optional parent-child cell overlap volume information.

    If not present, then the CellOverlap data-object lists the overlaps,
    but with no additional information.

    Parameters
    ----------
    volume_uom
        Units of measure for the overlapVolume.
    overlap_volumes
        Parent-child cell volume overlap. BUSINESS RULE: Length of array
        must equal the cell overlap count.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    volume_uom: Optional[VolumeUom] = field(
        default=None,
        metadata={
            "name": "VolumeUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    overlap_volumes: Optional[AbstractDoubleArray] = field(
        default=None,
        metadata={
            "name": "OverlapVolumes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ParameterTemplate:
    """
    Description of one parameter that participate in one type of activity.

    Parameters
    ----------
    key_constraint
        Allows you to indicate that, in the same activity, this parameter
        template must be associated to another parameter template identified
        by its title.
    is_input
        Indicates if the parameter is an input of the activity. If the
        parameter is a data object and is also an output of the activity, it
        is strongly advised to use two parameters : one for input and one
        for output. The reason is to be able to give two different versions
        strings for the input and output dataobject which has got obviously
        the same UUID.
    allowed_kind
        If no allowed type is given, then all kind of datatypes is allowed.
    is_output
        Indicates if the parameter is an output of the activity. If the
        parameter is a data object and is also an input of the activity, it
        is strongly advised to use two parameters : one for input and one
        for output. The reason is to be able to give two different versions
        strings for the input and output dataobject which has got obviously
        the same UUID.
    title
        Name of the parameter in the activity. Key to identify parameter.
    data_object_content_type
        When parameter is limited to data object of given types, describe
        the allowed types. Used only when ParameterType is dataObject
    max_occurs
        Maximum number of parameters of this type allowed in the activity.
        -1 means "infinite".
    min_occurs
        Minimum number of parameters of this type required by the activity.
        -1 means "infinite".
    constraint
        Textual description of additional constraint associated with the
        parameter. (note that it will be better to have a formal description
        of the constraint)
    default_value
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    is_input: bool = field(
        metadata={
            "name": "IsInput",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )

    is_output: bool = field(
        metadata={
            "name": "IsOutput",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    title: str = field(
        metadata={
            "name": "Title",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    max_occurs: int = field(
        metadata={
            "name": "MaxOccurs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    min_occurs: int = field(
        metadata={
            "name": "MinOccurs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    key_constraint: List[str] = field(
        default_factory=list,
        metadata={
            "name": "KeyConstraint",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    allowed_kind: List[ParameterKind] = field(
        default_factory=list,
        metadata={
            "name": "AllowedKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    data_object_content_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "DataObjectContentType",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

    constraint: Optional[str] = field(
        default=None,
        metadata={
            "name": "Constraint",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    default_value: List[AbstractActivityParameter] = field(
        default_factory=list,
        metadata={
            "name": "DefaultValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ParametricLineIntersections:
    """Used to specify the intersections between parametric lines.

    This information is purely geometric and is not required for the
    evaluation of the parametric point locations on these lines. The
    information required for that purpose is stored in the parametric
    points array.

    Parameters
    ----------
    count
        Number of parametric line intersections. Must be positive.
    intersection_line_pairs
        Intersected line index pair for (line 1, line 2). Size = 2 x count
    parameter_value_pairs
        Intersected line parameter value pairs for (line 1, line 2). Size =
        2 x count
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    intersection_line_pairs: AbstractIntegerArray = field(
        metadata={
            "name": "IntersectionLinePairs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parameter_value_pairs: AbstractValueArray = field(
        metadata={
            "name": "ParameterValuePairs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point2dHdf5Array(AbstractPoint3dArray):
    """An array of explicit XY points stored as two coordinates in an HDF5 dataset.

    If needed, the implied Z coordinate is uniformly 0.

    Parameters
    ----------
    coordinates
        Reference to an HDF5 2D dataset of XY points. The 2 coordinates are
        stored sequentially in HDF5, i.e., a multi-dimensional array of
        points is stored as a 2 x ... HDF5 array.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    coordinates: Hdf5Dataset = field(
        metadata={
            "name": "Coordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3dHdf5Array(AbstractPoint3dArray):
    """
    N array of explicit XYZ points stored as three coordinates in an HDF5 dataset.

    Parameters
    ----------
    coordinates
        Reference to an HDF5 3D dataset of XYZ points. The 3 coordinates are
        stored sequentially in HDF5, i.e., a multi-dimensional array of
        points is stored as a 3 x ... HDF5 array.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    coordinates: Hdf5Dataset = field(
        metadata={
            "name": "Coordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3dOffset:
    """Defines the size and sampling in each dimension (direction) of the point 3D
    lattice array.

    Sampling can be uniform or irregular.

    Parameters
    ----------
    offset
        The direction of the axis of this lattice dimension. This is a
        relative offset vector instead of an absolute 3D point.
    spacing
        A lattice of N offset points is described by a spacing array of size
        N-1. The offset between points is given by the spacing value
        multiplied by the offset vector. For example, the first offset is 0.
        The second offset is the first spacing * offset. The second offset
        is (first spacing + second spacing) * offset, etc.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    offset: Point3d = field(
        metadata={
            "name": "Offset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    spacing: AbstractDoubleArray = field(
        metadata={
            "name": "Spacing",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3dParametricArray(AbstractPoint3dArray):
    """
    A parametric specification of an array of XYZ points.

    Parameters
    ----------
    parameters
        A multi-dimensional array of parametric values that implicitly
        specifies an array of XYZ points. The parametric values provided in
        this data-object must be consistent with the parametric values
        specified in the referenced parametric line array. When constructing
        a column-layer grid geometry using parametric points, the array
        indexing follows the dimensionality of the coordinate lines x NKL,
        which is either a 2D or 3D array.
    parametric_line_indices
        An optional array of indices that map from the array index to the
        index of the corresponding parametric line. If this information is
        known from context, then this array is not needed. For example, in
        either of these cases: (1) If the mapping from array index to
        parametric line is 1:1. (2) If the mapping has already been
        specified, as with the pillar Index from the column-layer geometry
        of a grid. For example, when constructing a column-layer grid
        geometry using parametric lines, the array indexing follows the
        dimensionality of the coordinate lines.
    truncated_line_indices
        A 2D array of line indices for use with intersecting parametric
        lines. Each record consists of a single line index, which indicates
        the array line that uses this truncation information, followed by
        the parametric line indices for each of the points on that line. For
        a non-truncated line, the equivalent record repeats the array line
        index NKL+1 times. Size = (NKL+1) x truncatedLineCount
    parametric_lines
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    parameters: AbstractValueArray = field(
        metadata={
            "name": "Parameters",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parametric_lines: AbstractParametricLineArray = field(
        metadata={
            "name": "ParametricLines",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parametric_line_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "ParametricLineIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    truncated_line_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "TruncatedLineIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3dZValueArray(AbstractPoint3dArray):
    """An array of points defined by applying a Z value on top of an existing array
    of points, XYZ, where Z is ignored. Used in these cases:

    - in 2D for defining geometry of one patch of a 2D grid representation.
    - for extracting nodal geometry from one grid representation for use in another.

    Parameters
    ----------
    supporting_geometry
        Geometry defining the X and Y coordinates.
    zvalues
        The values for Z coordinates
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    supporting_geometry: AbstractPoint3dArray = field(
        metadata={
            "name": "SupportingGeometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    zvalues: AbstractDoubleArray = field(
        metadata={
            "name": "ZValues",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ResqmlJaggedArray:
    """Representation for an array of 1D variable length arrays. The representation
    consists of these two arrays:

    - An aggregation of all the variable length arrays into a single dimensional array.
    - The offsets into the other array, given by the sum of all the previous array lengths, including the current array.

    Parameters
    ----------
    elements
        1D array of elements containing the aggregation of individual array
        data.
    cumulative_length
        1D array of cumulative lengths to the end of the current array. This
        is also equal to the index of the next element, i.e., the index in
        the elements array, for which the current variable length array
        begins.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    elements: AbstractValueArray = field(
        metadata={
            "name": "Elements",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cumulative_length: AbstractIntegerArray = field(
        metadata={
            "name": "CumulativeLength",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Seismic2dCoordinates(AbstractSeismicCoordinates):
    """A group of 2D seismic coordinates that stores the 1-to-1 mapping between
    geometry patch coordinates (usually X, Y, Z) and trace or inter-trace positions
    on a seismic line.

    BUSINESS RULE: This patch must reference a geometry patch by its UUID.

    Parameters
    ----------
    line_abscissa
        The sequence of trace or inter-trace positions that correspond to
        the geometry coordinates. BUSINESS RULE: Both sequences must be in
        the same order.
    vertical_coordinates
        The sequence of vertical sample or inter-sample positions that
        correspond to the geometry coordinates. BUSINESS RULE: Sequence must
        be in the same order than previous one.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    line_abscissa: AbstractDoubleArray = field(
        metadata={
            "name": "LineAbscissa",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    vertical_coordinates: Optional[AbstractDoubleArray] = field(
        default=None,
        metadata={
            "name": "VerticalCoordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Seismic3dCoordinates(AbstractSeismicCoordinates):
    """
    The 1-to-1 mapping between geometry coordinates (usually X, Y, Z or X, Y, TWT)
    and trace or inter-trace positions on a seismic lattice.

    Parameters
    ----------
    crossline_coordinates
        The sequence of trace or inter-trace crossline positions that
        correspond to the geometry coordinates. BUSINESS RULE: Both
        sequences must be in the same order.
    inline_coordinates
        The sequence of trace or inter-trace inline positions that
        correspond to the geometry coordinates. BUSINESS RULE: Both
        sequences must be in the same order.
    vertical_coordinates
        The sequence of vertical sample or inter-sample positions that
        correspond to the geometry coordinates. BUSINESS RULE: Sequence must
        be in the same order than two previous ones.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    crossline_coordinates: AbstractDoubleArray = field(
        metadata={
            "name": "CrosslineCoordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    inline_coordinates: AbstractDoubleArray = field(
        metadata={
            "name": "InlineCoordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    vertical_coordinates: Optional[AbstractDoubleArray] = field(
        default=None,
        metadata={
            "name": "VerticalCoordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StreamlineWellbores:
    """
    The information that allows you to locate, on one or several grids (existing or
    planned), the intersection of volume (cells) and surface (faces) elements with
    a wellbore trajectory (existing or planned).

    Parameters
    ----------
    injector_per_line
        Size of array = LineCount. Null values of -1 signify that that line
        does not initiate at a injector, e.g., it may come from fluid
        expansion or an aquifer.
    producer_per_line
        Size of array = LineCount. Null values of -1 signify that that line
        does not terminate at a producer, e.g., it may approach a stagnation
        area. BUSINESS RULE: The cell count must equal the number of non-
        null entries in this array.
    wellbore_trajectory_representation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    injector_per_line: AbstractIntegerArray = field(
        metadata={
            "name": "InjectorPerLine",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    producer_per_line: AbstractIntegerArray = field(
        metadata={
            "name": "ProducerPerLine",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    wellbore_trajectory_representation: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "WellboreTrajectoryRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StringHdf5Array(AbstractValueArray):
    """Used to store explicit string values, i.e., values that are not double,
    boolean or integers.

    The datatype of the values will be identified by means of the HDF5
    API.

    Parameters
    ----------
    values
        Reference to HDF5 array of integer or double
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    values: Hdf5Dataset = field(
        metadata={
            "name": "Values",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StringParameter(AbstractActivityParameter):
    """
    Parameter containing a string value.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: str = field(
        default=None,
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeIndexParameter(AbstractActivityParameter):
    """
    Parameter containing a time index value.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    time_index: TimeIndex = field(
        default=None,
        metadata={
            "name": "TimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeIndexParameterKey(AbstractParameterKey):
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    time_index: TimeIndex = field(
        metadata={
            "name": "TimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeIndices:
    """Indices into a time series.

    Used to specify time. (Not to be confused with time step.)

    Parameters
    ----------
    time_index_count
    time_index_start
        The index of the start time in the time series, if not zero.
    simulator_time_step
        Simulation time step for each time index
    use_interval
        When UseInterval is true, the values are associated with each time
        intervals between two consecutive time entries instead of each
        individual time entry. As a consequence the dimension of the value
        array corresponding to the time series is the number of entry in the
        series minus one.
    time_series
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    time_index_count: int = field(
        metadata={
            "name": "TimeIndexCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    use_interval: bool = field(
        metadata={
            "name": "UseInterval",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    time_series: DataObjectReference = field(
        metadata={
            "name": "TimeSeries",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    time_index_start: Optional[int] = field(
        default=None,
        metadata={
            "name": "TimeIndexStart",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    simulator_time_step: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "SimulatorTimeStep",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeSeriesParentage:
    """
    Indicates that a time series has the associated time series as a parent, i.e.,
    that the series continues from the parent time series.

    Parameters
    ----------
    has_overlap
        Used to indicate that a time series overlaps with its parent time
        series, e.g., as may be done for simulation studies, where the end
        state of one calculation is the initial state of the next.
    parent_time_index
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    has_overlap: bool = field(
        metadata={
            "name": "HasOverlap",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_time_index: TimeIndex = field(
        metadata={
            "name": "ParentTimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UniformSubnodePatch(SubnodePatch):
    """
    Use this subnode construction if the number of subnodes is the same for every
    object, e.g., 3 subnodes per edge for all edges.

    Parameters
    ----------
    subnode_count_per_object
        Number of subnodes per object, with the same number for each of this
        object kind in the grid.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    subnode_count_per_object: int = field(
        metadata={
            "name": "SubnodeCountPerObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredGridHingeNodeFaces:
    """Hinge nodes define a triangulated interpolation on a cell face.

    In practice, they arise on the K faces of column layer cells and are used to add additional geometric resolution to the shape of the cell. The specification of triangulated interpolation also uniquely defines the interpolation schema on the cell face, and hence the cell volume.
    For an unstructured cell grid, the hinge node faces need to be defined explicitly.
    This hinge node faces object is optional and is only expected to be used if the hinge node faces higher order grid point attachment arises. Hinge node faces are not supported for property attachment. Instead use a subrepresentation or an attachment kind of faces or faces per cell.
    BUSINESS RULE: Each cell must have either 0 or 2 hinge node faces, so that the two hinge nodes for the cell may be used to define a cell center line and a cell thickness.

    Parameters
    ----------
    count
        Number of K faces. This count must be positive.
    face_indices
        List of faces to be identified as K faces for hinge node geometry
        attachment. BUSINESS RULE: Array length equals K face count.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    face_indices: AbstractIntegerArray = field(
        metadata={
            "name": "FaceIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VariableSubnodePatch(SubnodePatch):
    """
    If the number of subnodes per object are variable for each object, use this
    subnode construction.

    Parameters
    ----------
    object_indices
        Indices of the selected objects
    subnode_count_per_selected_object
        Number of subnodes per selected object.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    object_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ObjectIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    subnode_count_per_selected_object: AbstractIntegerArray = field(
        metadata={
            "name": "SubnodeCountPerSelectedObject",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VolumeRegion:
    """
    The volume within a shell or envelope.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_index: int = field(
        metadata={
            "name": "PatchIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )

    represents: DataObjectReference = field(
        metadata={
            "name": "Represents",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    external_shell: VolumeShell = field(
        metadata={
            "name": "ExternalShell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    internal_shells: List[VolumeShell] = field(
        default_factory=list,
        metadata={
            "name": "InternalShells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CharacterString_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gco"

    dq_evaluation_method_type_code: Optional[DQ_EvaluationMethodTypeCode] = field(
        default=None,
        metadata={
            "name": "DQ_EvaluationMethodTypeCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    ci_presentation_form_code: Optional[CI_PresentationFormCode] = field(
        default=None,
        metadata={
            "name": "CI_PresentationFormCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    ci_role_code: Optional[CI_RoleCode] = field(
        default=None,
        metadata={
            "name": "CI_RoleCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    ci_on_line_function_code: Optional[CI_OnLineFunctionCode] = field(
        default=None,
        metadata={
            "name": "CI_OnLineFunctionCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    ci_date_type_code: Optional[CI_DateTypeCode] = field(
        default=None,
        metadata={
            "name": "CI_DateTypeCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    character_string: Optional[str] = field(
        default=None,
        metadata={
            "name": "CharacterString",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gco",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractEX_GeographicExtent_Type(AbstractObject_Type):
    """
    Geographic area of the dataset.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    extent_type_code: Optional[Boolean_PropertyType] = field(
        default=None,
        metadata={
            "name": "extentTypeCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_DateTypeCode_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_date_type_code: Optional[CI_DateTypeCode] = field(
        default=None,
        metadata={
            "name": "CI_DateTypeCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_OnLineFunctionCode_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_on_line_function_code: Optional[CI_OnLineFunctionCode] = field(
        default=None,
        metadata={
            "name": "CI_OnLineFunctionCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_PresentationFormCode_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_presentation_form_code: Optional[CI_PresentationFormCode] = field(
        default=None,
        metadata={
            "name": "CI_PresentationFormCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_RoleCode_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_role_code: Optional[CI_RoleCode] = field(
        default=None,
        metadata={
            "name": "CI_RoleCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DQ_EvaluationMethodTypeCode_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    dq_evaluation_method_type_code: Optional[DQ_EvaluationMethodTypeCode] = field(
        default=None,
        metadata={
            "name": "DQ_EvaluationMethodTypeCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_TemporalExtent_Type(AbstractObject_Type):
    """
    Time period covered by the content of the dataset.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    extent: TM_Primitive_PropertyType = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_VerticalExtent_Type(AbstractObject_Type):
    """
    Vertical domain of dataset.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    minimum_value: Real_PropertyType = field(
        default=None,
        metadata={
            "name": "minimumValue",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    maximum_value: Real_PropertyType = field(
        default=None,
        metadata={
            "name": "maximumValue",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    vertical_crs: "SC_CRS_PropertyType" = field(
        default=None,
        metadata={
            "name": "verticalCRS",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RelatedTimeType(TimePrimitivePropertyType):
    """Gml:RelatedTimeType provides a content model for indicating the relative
    position of an arbitrary member of the substitution group whose head is
    gml:AbstractTimePrimitive.

    It extends the generic gml:TimePrimitivePropertyType with an XML
    attribute relativePosition, whose value is selected from the set of
    13 temporal relationships identified by Allen (1983)
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    relative_position: Optional[RelatedTimeType_relativePosition] = field(
        default=None,
        metadata={
            "name": "relativePosition",
            "type": "Attribute",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class axisDirection(CodeWithAuthorityType):
    """Gml:axisDirection is the direction of this coordinate system axis (or in the
    case of Cartesian projected coordinates, the direction of this coordinate
    system axis at the origin).

    Within any set of coordinate system axes, only one of each pair of
    terms may be used. For earth-fixed CRSs, this direction is often
    approximate and intended to provide a human interpretable meaning to
    the axis. When a geodetic datum is used, the precise directions of
    the axes may therefore vary slightly from this approximate
    direction. The codeSpace attribute shall reference a source of
    information specifying the values and meanings of all the allowed
    string values for this property.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class conversion(GeneralConversionPropertyType):
    """
    Gml:conversion is an association role to the coordinate conversion used to
    define the derived CRS.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class description(StringOrRefType):
    """The value of this property is a text description of the object.

    gml:description uses gml:StringOrRefType as its content model, so it
    may contain a simple text string content, or carry a reference to an
    external description. The use of gml:description to reference an
    external description has been deprecated and replaced by the
    gml:descriptionReference property.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class descriptionReference(ReferenceType):
    """The value of this property is a remote text description of the object.

    The xlink:href attribute of the gml:descriptionReference property
    references the external description.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class identifier(CodeWithAuthorityType):
    """Often, a special identifier is assigned to an object by the maintaining
    authority with the intention that it is used in references to the object For
    such cases, the codeSpace shall be provided.

    That identifier is usually unique either globally or within an
    application domain. gml:identifier is a pre-defined property for
    such identifiers.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class rangeMeaning(CodeWithAuthorityType):
    """Gml:rangeMeaning describes the meaning of axis value range specified by
    gml:minimumValue and gml:maximumValue.

    This element shall be omitted when both gml:minimumValue and
    gml:maximumValue are omitted. This element should be included when
    gml:minimumValue and/or gml:maximumValue are included. If this
    element is omitted when the gml:minimumValue and/or gml:maximumValue
    are included, the meaning is unspecified. The codeSpace attribute
    shall reference a source of information specifying the values and
    meanings of all the allowed string values for this property.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ProjectedCrs_1(AbstractCitedDataObject):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        name = "ProjectedCrs"
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    axis_order: AxisOrder2d = field(
        default=None,
        metadata={
            "name": "AxisOrder",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    abstract_projected_crs: AbstractProjectedCrs = field(
        default=None,
        metadata={
            "name": "AbstractProjectedCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    uom: Optional[LengthUom] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCrs_1(AbstractCitedDataObject):
    class Meta:
        name = "VerticalCrs"
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    direction: VerticalDirection = field(
        default=None,
        metadata={
            "name": "Direction",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    abstract_vertical_crs: AbstractVerticalCrs = field(
        default=None,
        metadata={
            "name": "AbstractVerticalCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )
    uom: LengthUom = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_EpcExternalPartReference(AbstractCitedDataObject):
    """It defines a proxy for external part of the EPC package.

    It must be used at least for external HDF parts.

    Parameters
    ----------
    mime_type
        IAMF registered, if one exists, or a free text field. Needs
        documentation on seismic especially. MIME type for HDF proxy is :
        application/x-hdf5 (by RESQML convention).
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    mime_type: str = field(
        default=None,
        metadata={
            "name": "MimeType",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractParametricLineGeometry(AbstractGeometry):
    """
    The abstract class for defining a single parametric line.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractPlaneGeometry(AbstractGeometry):
    """
    The abstract class for all geometric values defined by planes.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractResqmlDataObject(AbstractCitedDataObject):
    """The parent class for all top-level elements in RESQML.

    Inherits from AbstractCitedDataObject in the commonV2 package of the
    model.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    extra_metadata: List[NameValuePair] = field(
        default_factory=list,
        metadata={
            "name": "ExtraMetadata",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CellOverlap:
    """Optional cell volume overlap information between the current grid (the
    child) and the parent grid.

    Use this data-object when the child grid has an explicitly defined
    geometry, and these relationships cannot be inferred from the regrid
    descriptions.

    Parameters
    ----------
    count
        Number of parent-child cell overlaps. Must be positive.
    parent_child_cell_pairs
        (Parent cell index, Child cell index) pair for each overlap.
        BUSINESS RULE: Length of array must equal 2 x overlapCount.
    overlap_volume
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: Optional[int] = field(
        default=None,
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    parent_child_cell_pairs: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "ParentChildCellPairs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    overlap_volume: Optional[OverlapVolume] = field(
        default=None,
        metadata={
            "name": "OverlapVolume",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ColumnLayerSplitCoordinateLines:
    """Definition of the indexing for the split coordinate lines.

    When present, this indexing contributes to the coordinate line
    nodes.

    Parameters
    ----------
    count
        Number of split coordinate lines. The count must be positive.
    pillar_indices
        Pillar index for each split coordinate line. Length of this array is
        equal to the number of split coordinate lines. For the first
        pillarCount lines, the index of the coordinate line equals the index
        of the corresponding pillar.  This array provides the pillar indices
        for the additional (split) coordinate lines. Used to implicitly
        define column and cell geometry.
    columns_per_split_coordinate_line
        Column indices for each of the split coordinate lines. Used to
        implicitly define column and cell geometry. List-of-lists
        construction used to support shared coordinate lines.
    split_column_edges
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillar_indices: AbstractIntegerArray = field(
        metadata={
            "name": "PillarIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    columns_per_split_coordinate_line: ResqmlJaggedArray = field(
        metadata={
            "name": "ColumnsPerSplitCoordinateLine",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    split_column_edges: Optional[ColumnLayerSplitColumnEdges] = field(
        default=None,
        metadata={
            "name": "SplitColumnEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ConnectionInterpretations:
    """
    Parameters
    ----------
    interpretation_indices
        Indices for the interpretations for each connection, if any. The use
        of a Resqml jagged array allows zero or more than one interpretation
        to be associated with a single connection.
    feature_interpretation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    interpretation_indices: ResqmlJaggedArray = field(
        metadata={
            "name": "InterpretationIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    feature_interpretation: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "FeatureInterpretation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DoubleLatticeArray(AbstractDoubleArray):
    """Represents an array of doubles based on an origin and a multi-dimensional
    offset.

    The offset is based on a linearization of a multi-dimensional offset.
    If count(i) is the number of elements in the dimension i and offset(i) is the offset in the dimension i, then:
    globalOffsetInNDimension = startValue+ ni*offset(n) + n_1i*count(n)*offset(n-1) + .... + 0i*count(n)*count(n-1)*....count(1)*offset(0)

    Parameters
    ----------
    start_value
        Value representing the global start for the lattice.
    offset
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    start_value: float = field(
        metadata={
            "name": "StartValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    offset: List[DoubleConstantArray] = field(
        default_factory=list,
        metadata={
            "name": "Offset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IjSplitColumnEdges:
    """Used to construct the indices for the cell faces.

    For IJK grids with IJ gaps, the split column edge indices must be
    defined explicitly. Otherwise, column edges are not required to
    describe the lowest order grid geometry, but may be needed for
    higher order geometries or properties.

    Parameters
    ----------
    count
        Number of IJ split column edges in this grid. Must be positive.
    pillars_per_split_column_edge
        Definition of the split column edges in terms of the pillars per
        split column edge. Pillar count per edge is usually 2, but the list-
        of-lists construction is used to allow split column edges to be
        defined by more than 2 pillars.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillars_per_split_column_edge: ResqmlJaggedArray = field(
        metadata={
            "name": "PillarsPerSplitColumnEdge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IntegerLatticeArray(AbstractIntegerArray):
    """Represents an array of integers based on an origin and a multi-dimensional
    offset.

    The offset is based on a linearization of a multi-dimensional offset.
    If count(i) is the number of elements in the dimension i and offset(i) is the offset in the dimension i, then:
    globalOffsetInNDimension = startValue+ ni*offset(n) + n_1i*count(n)*offset(n-1) + .... + 0i*count(n)*count(n-1)*....count(1)*offset(0)

    Parameters
    ----------
    start_value
        Value representing the global start for the lattice: i.e., iStart +
        jStart*ni + kStart*ni*nj
    offset
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    start_value: int = field(
        metadata={
            "name": "StartValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    offset: List[IntegerConstantArray] = field(
        default_factory=list,
        metadata={
            "name": "Offset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class NodesPerCell:
    """Optional component of Unstructured Cell Finite Elements.

    The choice of node order per cell is important for effective use of the RESQML finite element representations. If you are working with an application with a particular node ordering per cell, be sure to specify the nodes in that order here, for ease of use.
    BUSINESS RULE: If cell subnodes are used for unstructured grids, then nodes per cell must be defined.

    Parameters
    ----------
    nodes_per_cell
        Defines an ordered list of nodes per cell.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    nodes_per_cell: ResqmlJaggedArray = field(
        metadata={
            "name": "NodesPerCell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class NonSealedContactRepresentationPart(AbstractContactRepresentationPart):
    """
    Defines a nonsealed contact representation, meaning that this contact
    representation is defined by a geometry.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    geometry: AbstractGeometry = field(
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    contact: List[ContactPatch] = field(
        default_factory=list,
        metadata={
            "name": "Contact",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ParametricLineArray(AbstractParametricLineArray):
    """Defines an array of parametric lines of multiple kinds.

    These are the documented parametric line kinds; see additional information below:
    0 = vertical
    1 = linear spline (piecewise linear)
    2 = natural cubic spline
    3 = cubic spline
    4 = Z linear cubic spline
    5 = minimum-curvature spline
    (-1) = null: no line
    If isBounded=true in the line definition, then any out of range parametric values in the parametric points are truncated to the first or last control point. Otherwise, the interpolant in the first or last interval is used as an extrapolating function.
    Special Cases:
    (1) Natural cubic splines with only two control points reduce to linear interpolation.
    (2) If required but not defined, tangent vectors at a spline knot are calculated from the control point data using a quadratic fit to the control point and the two adjacent control points (if internal) or, if at an edge, by a vanishing second derivative. This calculation reduces locally to a natural spline.
    (3) If not expected but provided, then extraneous information is to be ignored, e.g., tangent vectors for linear splines.
    Vertical:
    (1) Control points are (X,Y,-).
    (2) Parameter values are interpreted as depth =&gt; (X,Y,Z), where the depth to Z conversion depends on the vertical CRS direction.
    Piecewise Linear:
    (1) Control points are (P,X,Y,Z).
    (2) Piecewise interpolation in (X,Y,Z) as a linear function of P.
    Natural Cubic:
    (1) Control points are (P,X,Y,Z).
    (2) First and second derivatives at each knot are inferred from a quadratic fit to the two adjacent control points, if internal, or, if external knots, by specifying a vanishing second derivative.
    (3) Interpolating basis functions are obtained by specifying values and second derivatives at the knots.
    Cubic and Minimum-Curvature.
    (1) Control points are (P,X,Y,Z).
    (2) Tangent vectors are (P,TX,TY,TZ). Tangent vectors are defined as the derivative of position with respect to the parameter. If the parameter is arc-length, then the tangent vectors are unit vectors, but not otherwise.
    (3) Interpolating cubic basis functions obtained by specifying values and first derivatives at the knots.
    (4) Interpolating minimum-curvature basis functions obtained by a circular arc construction that is constrained by the knot data. This differs from the unconstrained "drilling" algorithm in which the knot locations are not independent but for which the parameter must be arc length.
    Z Linear Cubic:
    (1) (X,Y) follow a natural cubic spline and Z follows a linear spline.
    (2) Parametric values cannot be freely chosen but are instead defined to take the values of 0,,,.N for a line with N intervals, N+1 control points.
    (3) On export, to go from Z to P, the RESQML "software writer" first needs to determine the interval and then uses linearity in Z to determine P. For the control points, the P values are 0...N and for values of Z, other than the control points, intermediate values of P arise.
    (4) On import, a RESQML "software reader" converts from P to Z using piecewise linear interpolation, and from P to X and Y using natural cubic spline interpolation. Other than the differing treatment of Z from X and Y, these are completely generic interpolation algorithms.
    (5) The use of P instead of Z for interpolation allows support for over-turned reservoir structures and removes any apparent discontinuities in parametric derivatives at the spline knots.

    Parameters
    ----------
    control_point_parameters
        An optional array of explicit control point parameters for all of
        the control points on each of the parametric lines. Used only if
        control point parameters are present. The number of explicit control
        point parameters per line is given by the count of non-null
        parameters on each line. Described as a 1D array, the control point
        parameter array is divided into segments of length count, with null
        (NaN) values added to each segment to fill it up. Size = count x
        #Lines, e.g., 2D or 3D BUSINESS RULE: This count should be zero for
        vertical and Z linear cubic parametric lines. For all other
        parametric line kinds, there should be one control point parameter
        for each control point. NOTES: (1) Vertical parametric lines do not
        require control point parameters (2) Z linear cubic splines have
        implicitly defined parameters. For a line with N intervals (N+1
        control points), the parametric values are P=0,...,N. BUSINESS RULE:
        The parametric values must be strictly monotonically increasing on
        each parametric line.
    control_points
        An array of 3D points for all of the control points on each of the
        parametric lines. The number of control points per line is given by
        the count of non-null 3D points on each line. Described as a 1D
        array, the control point array is divided into segments of length
        count, with null (NaN) values added to each segment to fill it up.
        Size = count x #Lines, e.g., 2D or 3D
    knot_count
        The first dimension of the control point, control point parameter,
        and tangent vector arrays for the parametric splines. The Knot Count
        is typically chosen to be the maximum number of control points,
        parameters or tangent vectors on any parametric line in the array of
        parametric lines.
    line_kind_indices
        An array of integers indicating the parametric line kind. 0 =
        vertical 1 = linear spline 2 = natural cubic spline 3 = cubic spline
        4 = Z linear cubic spline 5 = minimum-curvature spline (-1) = null:
        no line Size = #Lines, e.g., (1D or 2D)
    tangent_vectors
        An optional array that is of tangent vectors for all of the control
        points on each of the cubic and minimum-curvature parametric lines.
        Used only if tangent vectors are present. The number of tangent
        vectors per line is given by the count of non-null tangent vectors
        on each of these line kinds. Described as a 1D array, the tangent
        vector array is divided into segments of length count, with null
        (NaN) values added to each segment to fill it up. Size = count x
        #Lines, e.g., 2D or 3D BUSINESS RULE: For all lines other than the
        cubic and minimum-curvature parametric lines, this count is zero.
        For these line kinds, there is one tangent vector for each control
        point. If a tangent vector is missing, then it is computed in the
        same fashion as for a natural cubic spline. Specifically, to obtain
        the tangent at internal knots, the control points are fit by a
        quadratic function with the two adjacent control points. At edge
        knots, the second derivative vanishes.
    parametric_line_intersections
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    control_points: AbstractPoint3dArray = field(
        metadata={
            "name": "ControlPoints",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    knot_count: int = field(
        metadata={
            "name": "KnotCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    line_kind_indices: AbstractIntegerArray = field(
        metadata={
            "name": "LineKindIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    control_point_parameters: Optional[AbstractDoubleArray] = field(
        default=None,
        metadata={
            "name": "ControlPointParameters",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    tangent_vectors: Optional[AbstractPoint3dArray] = field(
        default=None,
        metadata={
            "name": "TangentVectors",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    parametric_line_intersections: Optional[ParametricLineIntersections] = field(
        default=None,
        metadata={
            "name": "ParametricLineIntersections",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PatchOfGeometry:
    """
    Indicates which patch of the representation has a new geometry.

    Parameters
    ----------
    representation_patch_index
        Patch index for the geometry attachment, if required
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    geometry: AbstractGeometry = field(
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation_patch_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "RepresentationPatchIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3dLatticeArray(AbstractPoint3dArray):
    """Describes a lattice array of points obtained by sampling from along a multi-
    dimensional lattice.

    Each dimension of the lattice can be uniformly or irregularly
    spaced.

    Parameters
    ----------
    all_dimensions_are_orthogonal
        The optional element that indicates that the offset vectors for each
        direction are mutually orthogonal to each other. This meta-
        information is useful to remove any doubt of orthogonality in case
        of numerical precision issues. BUSINESS RULE: If you don't know it
        or if only one lattice dimension is given, do not provide this
        element.
    origin
        The origin location of the lattice given as XYZ coordinates.
    offset
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    origin: Point3d = field(
        metadata={
            "name": "Origin",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    offset: List[Point3dOffset] = field(
        default_factory=list,
        metadata={
            "name": "Offset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    all_dimensions_are_orthogonal: Optional[bool] = field(
        default=None,
        metadata={
            "name": "AllDimensionsAreOrthogonal",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PointGeometry(AbstractGeometry):
    """
    The geometry of a set of points defined by their location in the local CRS,
    with optional seismic coordinates.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    points: AbstractPoint3dArray = field(
        default=None,
        metadata={
            "name": "Points",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    seismic_coordinates: Optional[AbstractSeismicCoordinates] = field(
        default=None,
        metadata={
            "name": "SeismicCoordinates",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Regrid:
    """One-dimensional I, J, or K refinement and coarsening regrid specification.

    The regrid description is organized using intervals. Within each
    interval, the number of parent and child cells is specified. Parent
    and child grid cell faces are aligned at interval boundaries. By
    default, child cells are uniformly sized within an interval unless
    weights are used to modify their size. If the child grid is a root
    grid with an independent geometry, then there will usually be only a
    single interval for a regrid, because internal cell faces are not
    necessarily aligned.

    Parameters
    ----------
    initial_index_on_parent_grid
        0-based index for the placement of the window on the parent grid.
    intervals
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    initial_index_on_parent_grid: int = field(
        metadata={
            "name": "InitialIndexOnParentGrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    intervals: Optional[Intervals] = field(
        default=None,
        metadata={
            "name": "Intervals",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SealedContactRepresentationPart(AbstractContactRepresentationPart):
    """Sealed contact elements that indicate that 2 or more contact patches are
    partially or totally colocated or equivalent.

    For possible types of identity, see IdentityKind.

    Parameters
    ----------
    identical_node_indices
        Indicate which nodes (identified by their common index in all
        contact patches) of the contact patches are identical. If this list
        is not present, then it indicates that all nodes in each
        representation are identical, on an element-by-element level.
    identity_kind
    contact
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    identity_kind: IdentityKind = field(
        metadata={
            "name": "IdentityKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    identical_node_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "IdenticalNodeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

    contact: List[ContactPatch] = field(
        default_factory=list,
        metadata={
            "name": "Contact",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 2,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SplitEdges:
    """If split nodes are used in the construction of a column layer grid and
    indexable elements of kind edges are referenced, then the grid edges need to be
    re-defined.

    Use Case: finite elements, especially for higher order geometry.

    Parameters
    ----------
    count
        Number of edges. Must be positive.
    parent_edge_indices
        Parent unsplit edge index for each of the additional split edges.
    faces_per_split_edge
        Association of faces with the split edges, used to infer continuity
        of property, geometry, or interpretation with an attachment kind of
        edges.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_edge_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ParentEdgeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    faces_per_split_edge: ResqmlJaggedArray = field(
        metadata={
            "name": "FacesPerSplitEdge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StreamlinePolylineSetPatch(Patch):
    """A patch containing a set of polylines. For performance reasons, the geometry
    of each patch is described in only one 1D array of 3D points, which aggregates
    the nodes of all the polylines together. To be able to separate the polyline
    descriptions, additional information is added about the type of each polyline
    (closed or not) and the number of 3D points (node count) of each polyline.

    This additional information is contained in two arrays which are associated with each polyline set patch. The dimension of these arrays is the number of polylines gathered in one polyline set patch.
    - The first array contains a Boolean for each polyline (closed or not closed)
    - The second array contains the count of nodes for each polyline.

    Parameters
    ----------
    node_count
        Total number of nodes. BUSINESS RULE: Should be equal to the sum of
        the number of nodes per polyline
    interval_count
        Total number of intervals. BUSINESS RULE: Should be equal to the sum
        of the count of intervals per polyline.
    closed_polylines
        Indicates whether a polyline is closed. If closed, then the interval
        count for that polyline is equal to the node count. If open, then
        the interval count for that polyline is one less than the node
        count.
    node_count_per_polyline
        The first number in the list defines the node count for the first
        polyline in the polyline set patch. The second number in the list
        defines the node count for the second polyline in the polyline set
        patch. etc.
    interval_grid_cells
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    node_count: int = field(
        metadata={
            "name": "NodeCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    interval_count: int = field(
        metadata={
            "name": "IntervalCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    closed_polylines: AbstractBooleanArray = field(
        metadata={
            "name": "ClosedPolylines",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    node_count_per_polyline: AbstractIntegerArray = field(
        metadata={
            "name": "NodeCountPerPolyline",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    interval_grid_cells: Optional[IntervalGridCells] = field(
        default=None,
        metadata={
            "name": "IntervalGridCells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SubRepresentationPatch(Patch1d):
    """Each sub-representation patch has its own list of representation indices,
    drawn from the supporting representation.

    If a list of pairwise elements is required, use two representation
    indices. The count of elements is defined in SubRepresenationPatch.
    Optional additional grid topology is available for grid
    representations.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    element_indices: List[ElementIndices] = field(
        default_factory=list,
        metadata={
            "name": "ElementIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
            "max_occurs": 2,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SubnodeTopology:
    """
    Finite element subnode topology for an unstructured cell can be either variable
    or uniform, but not columnar.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    variable_subnodes: List[VariableSubnodePatch] = field(
        default_factory=list,
        metadata={
            "name": "VariableSubnodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    uniform_subnodes: List[UniformSubnodePatch] = field(
        default_factory=list,
        metadata={
            "name": "UniformSubnodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TruncationCellPatch(Patch):
    """Truncation definitions for the truncated and split cells.

    BUSINESS RULE: Patch Index must be positive since a patch index of 0 refers to the fundamental column layer coordinate line nodes and cells.

    Parameters
    ----------
    truncation_node_count
        Number of additional nodes required for the truncation construction.
        Must be positive. Uses a separate enumeration and does not increase
        the number of nodes, except as noted below.
    truncation_face_count
        Number of additional faces required for the split and truncation
        construction. The construction does not modify existing face
        definitions, but instead uses these new faces to redefine the
        truncated cell geometry. Must be positive. These faces are added to
        the enumeration of faces for the grid
    truncation_cell_count
        Number of polyhedral cells created by truncation. Must be positive.
        Note: Parent cells are replace
    nodes_per_truncation_face
        Definition of the truncation faces is in terms of an ordered list of
        nodes. Node indexing is EXTENDED, i.e., is based on the list of
        untruncated grid nodes (always first) plus the split nodes (if any)
        and the truncation face nodes. Relative order of split nodes and
        truncation face nodes is set by the pillar indices.
    parent_cell_indices
        Parent cell index for each of the truncation cells. BUSINESS RULE:
        Size must match truncationCellCount
    local_faces_per_cell
        Local cell face index for those faces which are retained from the
        parent cell in the definition of the truncation cell. The use of a
        local cell face index, e.g., 0...5 for an IJK cell, can be used even
        if the face indices have not been defined.
    truncation_faces_per_cell
        Truncation face index for the additional cell faces which are
        required to complete the definition of the truncation cell. The
        resulting local cell face index follows the local faces per cell
        list, followed by the truncation faces in the order within the list-
        of-lists constructions.
    truncation_cell_face_is_right_handed
        Boolean mask used to indicate which truncation cell faces have an
        outwardly directed normal, following a right hand rule. Data size
        and order follows the truncationFacesPerCell list-of-lists.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    truncation_node_count: int = field(
        metadata={
            "name": "TruncationNodeCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    truncation_face_count: int = field(
        metadata={
            "name": "TruncationFaceCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    truncation_cell_count: int = field(
        metadata={
            "name": "TruncationCellCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    nodes_per_truncation_face: ResqmlJaggedArray = field(
        metadata={
            "name": "NodesPerTruncationFace",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_cell_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ParentCellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    local_faces_per_cell: ResqmlJaggedArray = field(
        metadata={
            "name": "LocalFacesPerCell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    truncation_faces_per_cell: ResqmlJaggedArray = field(
        metadata={
            "name": "TruncationFacesPerCell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    truncation_cell_face_is_right_handed: AbstractBooleanArray = field(
        metadata={
            "name": "TruncationCellFaceIsRightHanded",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredColumnEdges:
    """Column edges are used to construct the index for faces.

    For unstructured column layer grids, the column edge indices must be
    defined explicitly. Column edges are not required to describe lowest
    order grid geometry, but may be needed for higher order geometries
    or properties.

    Parameters
    ----------
    count
        Number of unstructured column edges in this grid. Must be positive.
    pillars_per_column_edge
        Definition of the column edges in terms of the pillars per column
        edge. Pillar count per edge is usually 2, but the list-of-lists
        construction is used to allow column edges to be defined by more
        than 2 pillars.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillars_per_column_edge: ResqmlJaggedArray = field(
        metadata={
            "name": "PillarsPerColumnEdge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Address_Type(AbstractObject_Type):
    """
    Location of the responsible individual or organisation.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    delivery_point: List[CharacterString_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "deliveryPoint",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    city: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    administrative_area: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "administrativeArea",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    postal_code: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "postalCode",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    country: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    electronic_mail_address: List[CharacterString_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "electronicMailAddress",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Date_Type(AbstractObject_Type):
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    date: Date_PropertyType = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    date_type: CI_DateTypeCode_PropertyType = field(
        default=None,
        metadata={
            "name": "dateType",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_OnlineResource_Type(AbstractObject_Type):
    """
    Information about online sources from which the dataset, specification, or
    community profile name and extended metadata elements can be obtained.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    linkage: URL_PropertyType = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    protocol: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    application_profile: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "applicationProfile",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    name: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    description: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    function: Optional[CI_OnLineFunctionCode_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Series_Type(AbstractObject_Type):
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    name: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    issue_identification: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "issueIdentification",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    page: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Telephone_Type(AbstractObject_Type):
    """
    Telephone numbers for contacting the responsible individual or organisation.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    voice: List[CharacterString_PropertyType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    facsimile: List[CharacterString_PropertyType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_TemporalExtent(EX_TemporalExtent_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_VerticalExtent(EX_VerticalExtent_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGMLType:
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"
    id: str = field(
        metadata={
            "type": "Attribute",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    description: Optional[description] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    description_reference: Optional[descriptionReference] = field(
        default=None,
        metadata={
            "name": "descriptionReference",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    identifier: Optional[identifier] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    name: List[name] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EpcExternalPartReference(obj_EpcExternalPartReference):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/commonv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractFeature(AbstractResqmlDataObject):
    """Something that has physical existence at some point during the exploration,
    development, production or abandonment of a reservoir.

    For example: It can be a boundary, a rock volume, a basin area, but also extends to a drilled well, a drilling rig, an injected or produced fluid, or a 2D, 3D, or 4D seismic survey.
    Features are divided into these categories: geologic or technical.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractFeatureInterpretation(AbstractResqmlDataObject):
    """
    The main class that contains all of the other feature interpretations included
    in this interpreted model.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    domain: Domain = field(
        default=None,
        metadata={
            "name": "Domain",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    interpreted_feature: DataObjectReference = field(
        default=None,
        metadata={
            "name": "InterpretedFeature",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    has_occured_during: Optional[TimeInterval] = field(
        default=None,
        metadata={
            "name": "HasOccuredDuring",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGridGeometry(PointGeometry):
    """
    Grid geometry described by means of points attached to nodes and additional
    optional points which may be attached to other indexable elements of the grid
    representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    additional_grid_points: List[AdditionalGridPoints] = field(
        default_factory=list,
        metadata={
            "name": "AdditionalGridPoints",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractLocal3dCrs(AbstractResqmlDataObject):
    """Defines a local 2D+1D coordinate reference system, by translation and
    rotation, whose origin is located at the (X,Y,Z) Offset from the Projected and
    Vertical 2D+1D CRS. The units of measure in XY follow the Projected Crs. The
    units of measure of the third coordinate is determined in the depth or concrete
    type. ArealRotation is a plane angle.

    Defines a local 3D CRS is subject to the following restrictions:
    - The projected 2d CRS must have orthogonal axes
    - The vertical 1d CRS must be chosen so that it is orthogonal to the plane defined by the projected 2d CRS
    As a consequence of the definition:
    - The local CRS forms a Cartesian system of axes.
    - The local areal axes are in the plane of the projected system.
    - The local areal axes are orthogonal to each other.
    This 3D system is semantically equivalent to a compound CRS composed of a local 2D areal system and a local 1D vertical system.
    The labels associated with the axes on this local system are X, Y, Z or X, Y, T.
    The relative orientation of the local Y axis with respect to the local X axis is identical to that of the global axes.

    Parameters
    ----------
    yoffset
        The Y offset of the origin of the local areal axes relative to the
        projected CRS origin. The value MUST represent the second axis of
        the coordinate system. The unit of measure is defined by the unit of
        measure for the projected 2D CRS.
    zoffset
        The Z offset of the origin of the local vertical axis relative to
        the vertical CRS origin. According to CRS type (depth or time) it
        corresponds to the depth or time datum The value MUST represent the
        third axis of the coordinate system. The unit of measure is defined
        by the unit of measure for the vertical CRS.
    areal_rotation
        The rotation of the local Y axis relative to the projected Y axis. -
        A positive value indicates a clockwise rotation from the projected Y
        axis. - A negative value indicates a counter-clockwise rotation form
        the projected Y axis.
    projected_axis_order
        Defines the coordinate system axis order of the global projected CRS
        when the projected CRS is an unknown CRS, else it must be correspond
        to the axis order of the projected  CRS.
    projected_uom
        Unit of measure of the associated Projected CRS. When the projected
        CRS is not unknown, it must be the same than the unit defined by the
        Projected CRS.
    vertical_uom
        Unit of measure of the associated Vertical CRS. When the vertical
        CRS is not unknown, it must be the same than the unit defined by the
        Vertical CRS.
    xoffset
        The X location of the origin of the local areal axes relative to the
        projected CRS origin. The value MUST represent the first axis of the
        coordinate system. The unit of measure is defined by the unit of
        measure for the projected 2D CRS.
    zincreasing_downward
        Indicates that Z values correspond to depth values and are
        increasing downward, as opposite to elevation values increasing
        upward. When the vertical CRS is not an unknown, it must correspond
        to the axis orientation of the vertical CRS.
    vertical_crs
    projected_crs
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    yoffset: float = field(
        default=None,
        metadata={
            "name": "YOffset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    zoffset: float = field(
        default=None,
        metadata={
            "name": "ZOffset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    areal_rotation: PlaneAngleMeasure = field(
        default=None,
        metadata={
            "name": "ArealRotation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    projected_axis_order: AxisOrder2d = field(
        default=None,
        metadata={
            "name": "ProjectedAxisOrder",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    projected_uom: LengthUom = field(
        default=None,
        metadata={
            "name": "ProjectedUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    vertical_uom: LengthUom = field(
        default=None,
        metadata={
            "name": "VerticalUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    xoffset: float = field(
        default=None,
        metadata={
            "name": "XOffset",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    zincreasing_downward: bool = field(
        default=None,
        metadata={
            "name": "ZIncreasingDownward",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    vertical_crs: AbstractVerticalCrs = field(
        default=None,
        metadata={
            "name": "VerticalCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    projected_crs: AbstractProjectedCrs = field(
        default=None,
        metadata={
            "name": "ProjectedCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractParentWindow:
    """Parent window specification, organized according to the topology of the
    parent grid.

    In addition to a window specification, for grids with I, J, and/or K
    coordinates, the parentage construction includes a regridding
    description that covers grid refinement, coarsening, or any
    combination of the two.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_overlap: Optional[CellOverlap] = field(
        default=None,
        metadata={
            "name": "CellOverlap",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractProperty(AbstractResqmlDataObject):
    """Base class for storing all property values on representations, except
    current geometry location.

    Values attached to a given element can be either a scalar or a
    vector. The size of the vector is constant on all elements, and it
    is assumed that all elements of the vector have identical property
    types and share the same unit of measure.

    Parameters
    ----------
    count
        Number of elements in a 1D list of properties. When used in a multi-
        dimensional array, count is always the fastest.
    indexable_element
    realization_index
        Optional element indicating the realization index (metadata). Used
        if the property is the result of a multi-realization process.
    time_step
        Indicates that the property is the output of a specific time step
        from a flow simulator. Time step is metadata that makes sense in the
        context of a specific simulation run, and should not be confused
        with the time index.
    time_index
    supporting_representation
    local_crs
    property_kind
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        default=None,
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    indexable_element: IndexableElements = field(
        default=None,
        metadata={
            "name": "IndexableElement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    realization_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "RealizationIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    time_step: Optional[int] = field(
        default=None,
        metadata={
            "name": "TimeStep",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    time_index: Optional[TimeIndex] = field(
        default=None,
        metadata={
            "name": "TimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    supporting_representation: DataObjectReference = field(
        default=None,
        metadata={
            "name": "SupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    local_crs: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "LocalCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    property_kind: AbstractPropertyKind = field(
        default=None,
        metadata={
            "name": "PropertyKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractPropertyLookup(AbstractResqmlDataObject):
    """Generic representation of a property lookup table.

    Each derived element provides specific lookup methods for different
    data types.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractRepresentation(AbstractResqmlDataObject):
    """The parent class of all specialized digital descriptions, which may provide
    a representation of a feature interpretation or a technical feature. It may be
    either of these:

    - based on a topology and contains the geometry of this digital description.
    - based on the topology or the geometry of another representation.
    Not all representations require a defined geometry. For example, it is not required for block-centered grids or wellbore frames. For representations without geometry, a software writer may provide null (NaN) values in the local 3D CRS, which is mandatory.
    TimeIndex is provided to describe time-dependent geometry.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    represented_interpretation: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "RepresentedInterpretation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ColumnLayerSubnodeTopology(SubnodeTopology):
    """
    This data-object consists of the Unstructured Cell Finite Elements subnode
    topology plus the column subnodes.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    column_subnodes: List[ColumnSubnodePatch] = field(
        default_factory=list,
        metadata={
            "name": "ColumnSubnodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Grid2dPatch(Patch):
    """Patch representing a single 2D grid and its geometry.

    The FastestAxisCount and the SlowestAxisCount determine the indexing
    of this grid 2D patch, by defining a one dimensional index for the
    2D grid as follows: Index = FastestIndex + FastestAxisCount *
    SlowestIndex This indexing order IS the data order when stored in
    HDF5, in which case, this would be a
    SlowestAxisCount*FastestAxisCount two dimensional array in HDF5.

    Parameters
    ----------
    fastest_axis_count
        The number of nodes in the fastest direction.
    slowest_axis_count
        The number of nodes in the slowest direction.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    fastest_axis_count: int = field(
        metadata={
            "name": "FastestAxisCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    slowest_axis_count: int = field(
        metadata={
            "name": "SlowestAxisCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: PointGeometry = field(
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class HorizontalPlaneGeometry(AbstractPlaneGeometry):
    """
    Defines the infinite geometry of a horizontal plane provided by giving its
    unique Z value.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    coordinate: float = field(
        default=None,
        metadata={
            "name": "Coordinate",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IjGaps:
    """
    Optional object used to indicate that adjacent columns of the model are split
    from each other, which is modeled by introducing additional (split) pillars.

    Parameters
    ----------
    split_pillar_count
        Number of split pillars in the model. Count must be positive.
    parent_pillar_indices
        Parent pillar index for each of the split pillars. This information
        is used to infer the grid cell geometry. BUSINESS RULE: Array length
        must match splitPillarCount.
    columns_per_split_pillar
        List of columns for each of the split pillars. This information is
        used to infer the grid cell geometry. BUSINESS RULE: The length of
        the first list-of-lists array must match the splitPillarCount.
    ij_split_column_edges
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    split_pillar_count: Optional[int] = field(
        default=None,
        metadata={
            "name": "SplitPillarCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    parent_pillar_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "ParentPillarIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    columns_per_split_pillar: Optional[ResqmlJaggedArray] = field(
        default=None,
        metadata={
            "name": "ColumnsPerSplitPillar",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    ij_split_column_edges: Optional[IjSplitColumnEdges] = field(
        default=None,
        metadata={
            "name": "IjSplitColumnEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class NodePatch(Patch1d):
    """
    Patch representing a list of nodes to which geometry may be attached.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    geometry: PointGeometry = field(
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ParametricLineFromRepresentationGeometry(AbstractParametricLineGeometry):
    """The parametric line extracted from an existing representation.

    BUSINESS RULE: The supporting representation must have pillars or lines as indexable elements.

    Parameters
    ----------
    line_indiex_on_supporting_representation
        The line index of the selected line in the supporting
        representation. For a column-layer grid, the parametric lines follow
        the indexing of the pillars.
    supporting_representation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    line_indiex_on_supporting_representation: int = field(
        default=None,
        metadata={
            "name": "LineIndiexOnSupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    supporting_representation: DataObjectReference = field(
        default=None,
        metadata={
            "name": "SupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ParametricLineFromRepresentationLatticeArray(AbstractParametricLineArray):
    """The lattice array of parametric lines extracted from an existing
    representation.

    BUSINESS RULE: The supporting representation must have pillars or lines as indexable elements.

    Parameters
    ----------
    line_indices_on_supporting_representation
        The line indices of the selected lines in the supporting
        representation. The index selection is regularly incremented from
        one node to the next node. BUSINESS RULE: The dimensions of the
        integer lattice array must be consistent with the dimensions of the
        supporting representation. For a column-layer grid, the parametric
        lines follow the indexing of the pillars. BUSINESS RULE: The start
        value of the integer lattice array must be the linearized index of
        the starting line. Example: iStart + ni * jStart in case of a
        supporting 2D grid.
    supporting_representation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    line_indices_on_supporting_representation: IntegerLatticeArray = field(
        metadata={
            "name": "LineIndicesOnSupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    supporting_representation: DataObjectReference = field(
        metadata={
            "name": "SupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ParametricLineGeometry(AbstractParametricLineGeometry):
    """Defines a parametric line of any kind.

    For more information on the supported parametric lines, see
    ParametricLineArray.

    Parameters
    ----------
    control_point_parameters
        An optional array of explicit control point parameters for the
        control points on the parametric line. Used only if control point
        parameters are present. NOTES: (1) Vertical parametric lines do not
        require control point parameters. (2) Z linear cubic splines have
        implicitly defined parameters. For a line with N intervals (N+1
        control points), the parametric values are P=0,...,N. BUSINESS RULE:
        If present, the size must match the number of control points.
        BUSINESS RULE: For vertical and Z linear cubic parametric lines,
        this count must be zero. For all other parametric line kinds, each
        control point must have one control point parameter. BUSINESS RULE:
        The parametric values must be strictly monotonically increasing on
        each parametric line. This is an optional array which should only be
        used if control point parameters are present. BUSINESS RILE: If
        present, the size must match the number of control points. BUSINESS
        RULE: This count should be zero for vertical and Z linear cubic
        parametric lines. For all other parametric line kinds there should
        be one control point parameter for each control point. Notes: (1)
        Vertical parametric lines do not require control point parameters
        (2) Z linear cubic splines have implicitly defined parameters. For a
        line with N intervals (N+1 control points), the parametric values
        are P=0,...,N. BUSINESS RULE: The parametric values must be strictly
        monotonically increasing on each parametric line.
    control_points
        An array of 3D points for the control points on the parametric line.
    knot_count
        Number of spline knots in the parametric line.
    line_kind_index
        Integer indicating the parametric line kind 0 for vertical 1 for
        linear spline 2 for natural cubic spline 3 for cubic spline 4 for z
        linear cubic spline 5 for minimum-curvature spline (-1) for null: no
        line
    tangent_vectors
        An optional array of tangent vectors for each control point on the
        cubic and minimum-curvature parametric lines. Used only if tangent
        vectors are present. If a tangent vector is missing, then it is
        computed in the same fashion as for a natural cubic spline.
        Specifically, to obtain the tangent at internal knots, the control
        points are fit by a quadratic function with the two adjacent control
        points. At edge knots, the second derivative vanishes.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    control_points: AbstractPoint3dArray = field(
        default=None,
        metadata={
            "name": "ControlPoints",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    knot_count: int = field(
        default=None,
        metadata={
            "name": "KnotCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    line_kind_index: int = field(
        default=None,
        metadata={
            "name": "LineKindIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    tangent_vectors: Optional[AbstractPoint3dArray] = field(
        default=None,
        metadata={
            "name": "TangentVectors",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    control_point_parameters: Optional[AbstractDoubleArray] = field(
        default=None,
        metadata={
            "name": "ControlPointParameters",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Point3dFromRepresentationLatticeArray(AbstractPoint3dArray):
    """A lattice array of points extracted from an existing representation.

    BUSINESS RULE: The supporting representation must have nodes as indexable elements

    Parameters
    ----------
    node_indices_on_supporting_representation
        The node indices of the selected nodes in the supporting
        representation. The index selection is regularly incremented from
        one node to the next node. BUSINESS RULE: The node indices must be
        consistent with the size of supporting representation.
    supporting_representation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    node_indices_on_supporting_representation: IntegerLatticeArray = field(
        metadata={
            "name": "NodeIndicesOnSupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    supporting_representation: DataObjectReference = field(
        metadata={
            "name": "SupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PolylineSetPatch(Patch):
    """A patch containing a set of polylines. For performance reasons, the geometry
    of each patch is described in only one 1D array of 3D points, which aggregates
    the nodes of all the polylines together. To be able to separate the polyline
    descriptions, additional information is added about the type of each polyline
    (closed or not) and the number of 3D points (node count) of each polyline.

    This additional information is contained in two arrays which are associated with each polyline set patch. The dimension of these arrays is the number of polylines gathered in one polyline set patch.
    - The first array contains a Boolean for each polyline (closed or not closed)
    - The second array contains the count of nodes for each polyline.

    Parameters
    ----------
    closed_polylines
    node_count_per_polyline
        The first number in the list defines the node count for the first
        polyline in the polyline set patch. The second number in the list
        defines the node count for the second polyline in the polyline set
        patch. etc.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    closed_polylines: AbstractBooleanArray = field(
        metadata={
            "name": "ClosedPolylines",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    node_count_per_polyline: AbstractIntegerArray = field(
        metadata={
            "name": "NodeCountPerPolyline",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: PointGeometry = field(
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SplitFaces:
    """Optional construction used to introduce additional faces created by split
    nodes.

    Used to represent complex geometries, e.g., for stair-step grids and
    reverse faults.

    Parameters
    ----------
    count
        Number of additional split faces. Count must be positive.
    parent_face_indices
        Parent unsplit face index for each of the additional split faces.
    cell_indices
        Cell index for each split face. Used to implicitly define cell
        geometry.
    split_edges
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_face_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ParentFaceIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cell_indices: AbstractIntegerArray = field(
        metadata={
            "name": "CellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    split_edges: Optional[SplitEdges] = field(
        default=None,
        metadata={
            "name": "SplitEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TiltedPlaneGeometry(AbstractPlaneGeometry):
    """
    Describes the geometry of a tilted (or potentially not tilted) plane from three
    points.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    plane: List[ThreePoint3d] = field(
        default_factory=list,
        metadata={
            "name": "Plane",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TrianglePatch(Patch1d):
    """Patch made of triangles, where the number of triangles is given by the patch count.
    BUSINESS RULE: Within a patch, all the triangles must be contiguous.
    The patch contains:
    - Number of nodes within the triangulation and their locations.
    - 2D array describing the topology of the triangles.
    Two triangles that are connected may be in different patches.

    Parameters
    ----------
    node_count
    triangles
        The triangles are a 2D array of non-negative integers with the
        dimensions 3 x numTriangles.
    geometry
    split_edge_patch
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    node_count: int = field(
        metadata={
            "name": "NodeCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    triangles: AbstractIntegerArray = field(
        metadata={
            "name": "Triangles",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: PointGeometry = field(
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    split_edge_patch: List[EdgePatch] = field(
        default_factory=list,
        metadata={
            "name": "SplitEdgePatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredSubnodeTopology(SubnodeTopology):
    """If edge subnodes are used, then edges must be defined.

    If cell subnodes are used, nodes per cell must be defined.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    edges: Optional[Edges] = field(
        default=None,
        metadata={
            "name": "Edges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    nodes_per_cell: Optional[NodesPerCell] = field(
        default=None,
        metadata={
            "name": "NodesPerCell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreMarker(AbstractResqmlDataObject):
    """Representation of a wellbore marker that is located along a wellbore
    trajectory, one for each MD value in the wellbore frame.

    BUSINESS RULE: Ordering of the wellbore markers must match the ordering of the nodes in the wellbore marker frame representation

    Parameters
    ----------
    fluid_contact
    fluid_marker
    geologic_boundary_kind
    witsml_formation_marker
        Optional WITSML wellbore reference of the well marker frame.
    interpretation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    fluid_contact: Optional[FluidContact] = field(
        default=None,
        metadata={
            "name": "FluidContact",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    fluid_marker: Optional[FluidMarker] = field(
        default=None,
        metadata={
            "name": "FluidMarker",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geologic_boundary_kind: Optional[GeologicBoundaryKind] = field(
        default=None,
        metadata={
            "name": "GeologicBoundaryKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    witsml_formation_marker: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "WitsmlFormationMarker",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    interpretation: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "Interpretation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_Activity(AbstractResqmlDataObject):
    """
    Instance of a given activity.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    activity_descriptor: DataObjectReference = field(
        default=None,
        metadata={
            "name": "ActivityDescriptor",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parameter: List[AbstractActivityParameter] = field(
        default_factory=list,
        metadata={
            "name": "Parameter",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    parent: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "Parent",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_ActivityTemplate(AbstractResqmlDataObject):
    """
    Description of one type of activity.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    parameter: List[ParameterTemplate] = field(
        default_factory=list,
        metadata={
            "name": "Parameter",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GlobalChronostratigraphicColumn(AbstractResqmlDataObject):
    """
    Chronological successions of some chronostratigraphic units organized into 1 to
    n chronological ranks.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    chronostratigraphic_column_component: List[ChronostratigraphicRank] = field(
        default_factory=list,
        metadata={
            "name": "ChronostratigraphicColumnComponent",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_LocalGridSet(AbstractResqmlDataObject):
    """Used to activate and/or deactivate the specified children grids as local
    grids on their parents.

    Once activated, this object indicates that a child grid replaces
    local portions of the corresponding parent grid. Parentage is
    inferred from the child grid construction. Without a grid set
    activation, the local grids are always active. Otherwise, the grid
    set activation is used to activate and/or deactivate the local grids
    in the set at specific times.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    activation: Optional[Activation] = field(
        default=None,
        metadata={
            "name": "Activation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    child_grid: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "ChildGrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_MdDatum(AbstractResqmlDataObject):
    """Specifies the location of the measured depth = 0 reference point.

    The location of this reference point is defined with respect to a
    CRS, which need not be the same as the CRS of a wellbore trajectory
    representation, which may reference this location.

    Parameters
    ----------
    location
        The location of the md reference point relative to a local CRS.
    md_reference
    local_crs
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    location: Point3d = field(
        default=None,
        metadata={
            "name": "Location",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    md_reference: MdReference = field(
        default=None,
        metadata={
            "name": "MdReference",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    local_crs: DataObjectReference = field(
        default=None,
        metadata={
            "name": "LocalCrs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PropertyKind(AbstractResqmlDataObject):
    """A description of a property name relative to a standard definition.

    For example, you may specify if the property kind is abstract, the
    dictionary in which the property is unique, and the representative
    unit of measure.

    Parameters
    ----------
    naming_system
        The name of the dictionary within which the property is unique. This
        also defines the name of the controlling authority. Use a URN of the
        form "urn:x-resqml:domainOrEmail:dictionaryName". An example public
        dictionary: "urn:resqml:energistics.org:RESQML" assigned to values
        defined by ResqmlPropertyKind. An example corporate dictionary:
        "urn:resqml:slb.com:product-x". An example personal dictionary:
        "urn:resqml:first.last@mycompany.com:my.first.dictionary". The
        purpose of this scheme is to generate a unique name. Parsing for
        semantics is not intended.
    is_abstract
        A value of true indicates that the property kind is abstract and an
        instance of property values must not represent this kind. A value of
        false indicates otherwise (i.e., that an instance of property values
        may represent this kind).
    representative_uom
        Generally matches the base for conversion, except where multiple
        classes have the same underlying dimensional analysis. In this case,
        the representative unit may provide additional information about the
        underlying concept of the class. For example, “area per volume” has
        the same dimensional analysis as “per length”, but it specifies a
        representative unit of “m2/m3” instead of “1/m”.
    parent_property_kind
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    naming_system: str = field(
        default=None,
        metadata={
            "name": "NamingSystem",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    is_abstract: bool = field(
        default=None,
        metadata={
            "name": "IsAbstract",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representative_uom: ResqmlUom = field(
        default=None,
        metadata={
            "name": "RepresentativeUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_property_kind: AbstractPropertyKind = field(
        default=None,
        metadata={
            "name": "ParentPropertyKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PropertySet(AbstractResqmlDataObject):
    """A set of properties collected together for a specific purpose.

    For example, a property set can be used to collect all the
    properties corresponding to the simulation output at a single time,
    or all the values of a single property type for all times.

    Parameters
    ----------
    time_set_kind
    has_single_property_kind
        If true, indicates that the collection contains only property values
        associated with a single property kind.
    has_multiple_realizations
        If true, indicates that the collection contains properties with
        defined realization indices.
    parent_set
        A pointer to the parent property group of this property group.
    properties
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    time_set_kind: TimeSetKind = field(
        default=None,
        metadata={
            "name": "TimeSetKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    has_single_property_kind: bool = field(
        default=None,
        metadata={
            "name": "HasSinglePropertyKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    has_multiple_realizations: bool = field(
        default=None,
        metadata={
            "name": "HasMultipleRealizations",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_set: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "ParentSet",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    properties: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Properties",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StratigraphicColumn(AbstractResqmlDataObject):
    """A global interpretation of the stratigraphy, which can be made up of several
    ranks of stratigraphic unit interpretations.

    BUSINESS RULE: All stratigraphic column rank interpretations that make up a stratigraphic column must be ordered by age.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    ranks: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Ranks",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_TimeSeries(AbstractResqmlDataObject):
    """Stores an ordered list of times, for example, for time-dependent properties,
    geometries, or representations.

    It is used in conjunction with the time index to specify times for
    RESQML.

    Parameters
    ----------
    time
        Individual times composing the series. The list ordering is used by
        the time index.
    time_series_parentage
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    time: List[Timestamp] = field(
        default_factory=list,
        metadata={
            "name": "Time",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    time_series_parentage: Optional[TimeSeriesParentage] = field(
        default=None,
        metadata={
            "name": "TimeSeriesParentage",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Address(CI_Address_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Date(CI_Date_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_OnlineResource(CI_OnlineResource_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Series(CI_Series_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Telephone(CI_Telephone_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_TemporalExtent_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ex_temporal_extent: Optional[EX_TemporalExtent] = field(
        default=None,
        metadata={
            "name": "EX_TemporalExtent",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_VerticalExtent_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ex_vertical_extent: Optional[EX_VerticalExtent] = field(
        default=None,
        metadata={
            "name": "EX_VerticalExtent",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractTimeObjectType(AbstractGMLType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DefinitionBaseType(AbstractGMLType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    identifier: identifier = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGeologicFeature(AbstractFeature):
    """Objects that exist a priori, in the natural world, for example: the rock
    formations and how they are positioned with regard to each other; the fluids
    that are present before production; or the position of the geological intervals
    with respect to each.

    Some of these objects are static—such as geologic intervals---while others are dynamic—such as fluids; their properties, geometries, and quantities may change over time during the course of field production.
    RESQML has these types of features: geologic and technical.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGridRepresentation(AbstractRepresentation):
    """
    Abstract class for all grid representations.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_fluid_phase_units: Optional[CellFluidPhaseUnits] = field(
        default=None,
        metadata={
            "name": "CellFluidPhaseUnits",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    parent_window: Optional[AbstractParentWindow] = field(
        default=None,
        metadata={
            "name": "ParentWindow",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    cell_stratigraphic_units: Optional[CellStratigraphicUnits] = field(
        default=None,
        metadata={
            "name": "CellStratigraphicUnits",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractOrganizationInterpretation(AbstractFeatureInterpretation):
    """The main class used to group features into meaningful units as a step in
    working towards the goal of building an earth model (the organization of all
    other organizations in RESQML).

    An organization interpretation:
    - Is typically comprised of one stack of its contained elements.
    - May be built on other organization interpretations.
    Typically contains:
    - contacts between the elements of this stack among themselves.
    - contacts between the stack elements and other organization elements.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    contact_interpretation: List[AbstractContactInterpretationPart] = field(
        default_factory=list,
        metadata={
            "name": "ContactInterpretation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractSurfaceRepresentation(AbstractRepresentation):
    """Parent class of structural surface representations, which can be bounded by
    an outer ring and has inner rings.

    These surfaces may consist of one or more patches.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    surface_role: SurfaceRole = field(
        default=None,
        metadata={
            "name": "SurfaceRole",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    boundaries: List[PatchBoundaries] = field(
        default_factory=list,
        metadata={
            "name": "Boundaries",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractTechnicalFeature(AbstractFeature):
    """Objects that exist by the action of humans.

    Examples include: wells and all they may contain, seismic surveys (surface, permanent water bottom), or injected fluid volumes. Because the decision to deploy such equipment is the result of studies or decisions by humans, technical features are usually not subject to the same kind of large changes in interpretation as geologic features. However, they are still subject to measurement error and other sources of uncertainty, and so still can be considered as subject to “interpretation”.
    RESQML has these types of features: geologic and technical.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractValuesProperty(AbstractProperty):
    """Base class for property values.

    Each derived element provides specific property values, including
    point property in support of geometries.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_of_values: List[PatchOfValues] = field(
        default_factory=list,
        metadata={
            "name": "PatchOfValues",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    facet: List[PropertyKindFacet] = field(
        default_factory=list,
        metadata={
            "name": "Facet",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Activity(obj_Activity):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ActivityTemplate(obj_ActivityTemplate):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CellParentWindow(AbstractParentWindow):
    """
    Parent window for ANY grid indexed as if it were an unstructured cell grid,
    i.e., using a 1D index.

    Parameters
    ----------
    cell_indices
        Cell indices which list the cells in the parent window. BUSINESS
        RULE: Number of cells must be consistent with the child grid cell
        count.
    parent_grid
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_indices: AbstractIntegerArray = field(
        default=None,
        metadata={
            "name": "CellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_grid: DataObjectReference = field(
        default=None,
        metadata={
            "name": "ParentGrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ColumnLayerParentWindow(AbstractParentWindow):
    """
    Parent window for any column layer grid indexed as if it were an unstructured
    column layer grid, i.e., IJ columns are replaced by a column index.

    Parameters
    ----------
    column_indices
        Column indices that list the columns in the parent window. BUSINESS
        RULE: Number of columns must be consistent with the child grid
        column count.
    omit_parent_cells
        List of parent cells that are to be retained at their original
        resolution and are not to be included within a local grid. The omit
        allows non-rectangular local grids to be specified. 0-based indexing
        follows #Columns x #Layers relative to the parent window cell count,
        not to the parent grid.
    kregrid
    parent_grid
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    column_indices: AbstractIntegerArray = field(
        default=None,
        metadata={
            "name": "ColumnIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    omit_parent_cells: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "OmitParentCells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    kregrid: Regrid = field(
        default=None,
        metadata={
            "name": "KRegrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_grid: DataObjectReference = field(
        default=None,
        metadata={
            "name": "ParentGrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GlobalChronostratigraphicColumn(obj_GlobalChronostratigraphicColumn):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IjkParentWindow(AbstractParentWindow):
    """
    Parent window for any IJK grid.

    Parameters
    ----------
    omit_parent_cells
        List of parent cells that are to be retained at their original
        resolution and are not to be included within a local grid. The
        "omit" allows non-rectangular local grids to be specified. 0-based
        indexing follows NI x NJ x NK relative to the parent window cell
        count—not to the parent grid.
    jregrid
    parent_grid
    kregrid
    iregrid
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    jregrid: Regrid = field(
        default=None,
        metadata={
            "name": "JRegrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_grid: DataObjectReference = field(
        default=None,
        metadata={
            "name": "ParentGrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    kregrid: Regrid = field(
        default=None,
        metadata={
            "name": "KRegrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    iregrid: Regrid = field(
        default=None,
        metadata={
            "name": "IRegrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )

    omit_parent_cells: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "OmitParentCells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LocalGridSet(obj_LocalGridSet):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MdDatum(obj_MdDatum):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PropertyKind(obj_PropertyKind):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PropertySet(obj_PropertySet):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SplitNodePatch(Patch):
    """Optional construction used to introduce additional nodes on coordinate
    lines.

    Used to represent complex geometries, e.g., for stair-step grids and reverse faults.
    BUSINESS RULE: Patch Index must be positive since a patch index of 0 refers to the fundamental column layer coordinate line nodes.

    Parameters
    ----------
    count
        Number of additional split nodes. Count must be positive.
    parent_node_indices
        Parent coordinate line node index for each of the split nodes. Used
        to implicitly define cell geometry.
    cells_per_split_node
        Cell indices for each of the split nodes. Used to implicitly define
        cell geometry. List-of-lists construction used to support split
        nodes shared between multiple cells.
    split_faces
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    parent_node_indices: AbstractIntegerArray = field(
        metadata={
            "name": "ParentNodeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cells_per_split_node: ResqmlJaggedArray = field(
        metadata={
            "name": "CellsPerSplitNode",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    split_faces: Optional[SplitFaces] = field(
        default=None,
        metadata={
            "name": "SplitFaces",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StratigraphicColumn(obj_StratigraphicColumn):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TimeSeries(obj_TimeSeries):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredGridGeometry(AbstractGridGeometry):
    """Description of the geometry of an unstructured cell grid, which includes
    geometric characteristics, e.g., cell face parity, and supporting topology.

    Each grid cell is defined by a (signed) list of cell faces. Each
    cell face is defined by a list of nodes.

    Parameters
    ----------
    cell_shape
    node_count
        Total number of nodes in the grid. Must be positive.
    face_count
        Total number of faces in the grid. Must be positive.
    nodes_per_face
        List of nodes per face. node count per face can be obtained from the
        offsets in the first list of list array. BUSINESS RULE: faceCount
        must match the length of the first list of list array.
    faces_per_cell
        List of faces per cell. face count per cell can be obtained from the
        offsets in the first list of list array. BUSINESS RULE: cellCount
        must match the length of the first list of list array.
    cell_face_is_right_handed
        Boolean mask used to indicate which cell faces have an outwardly
        directed normal following a right hand rule. Array length is the sum
        of the cell face count per cell, and the data follows the order of
        the faces per cell resqml list-of-lists.
    hinge_node_faces
    subnode_topology
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_shape: CellShape = field(
        default=None,
        metadata={
            "name": "CellShape",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    node_count: int = field(
        default=None,
        metadata={
            "name": "NodeCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    face_count: int = field(
        default=None,
        metadata={
            "name": "FaceCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    nodes_per_face: ResqmlJaggedArray = field(
        default=None,
        metadata={
            "name": "NodesPerFace",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    faces_per_cell: ResqmlJaggedArray = field(
        default=None,
        metadata={
            "name": "FacesPerCell",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cell_face_is_right_handed: AbstractBooleanArray = field(
        default=None,
        metadata={
            "name": "CellFaceIsRightHanded",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    hinge_node_faces: Optional[UnstructuredGridHingeNodeFaces] = field(
        default=None,
        metadata={
            "name": "HingeNodeFaces",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    subnode_topology: Optional[UnstructuredSubnodeTopology] = field(
        default=None,
        metadata={
            "name": "SubnodeTopology",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_BoundaryFeatureInterpretation(AbstractFeatureInterpretation):
    """The main class for data describing an opinion of a surface feature between
    two volumes.

    BUSINESS RULE: The data-object reference (of type "interprets") must reference only a boundary feature.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_DeviationSurveyRepresentation(AbstractRepresentation):
    """Specifies the station data from a deviation survey.

    The deviation survey does not provide a complete specification of
    the geometry of a wellbore trajectory. Although a minimum-curvature
    algorithm is used in most cases, the implementation varies
    sufficiently that no single algorithmic specification is available
    as a data transfer standard. Instead, the geometry of a RESQML
    wellbore trajectory is represented by a parametric line,
    parameterized by the MD. CRS and units of measure do not need to be
    consistent with the CRS and units of measure for wellbore trajectory
    representation.

    Parameters
    ----------
    witsml_deviation_survey
    is_final
        Used to indicate that this is a final version of the deviation
        survey, as distinct from the interim interpretations.
    station_count
        Number of Stations
    md_uom
        Units of Measure of the measured depths along this deviation survey.
    mds
        MD values for the position of the stations BUSINESS RULE: Array
        length equals station count
    first_station_location
        XYZ location of the first station of the deviation survey.
    angle_uom
        Defines the units of measure for the azimuth and inclination
    azimuths
        An array of azimuth angles, one for each survey station. The
        rotation is relative to the ProjectedCrs north with a positive value
        indication a clockwise rotation as seen from above. If the local CRS
        - whether a LocalTime3dCrs or a LocalDepth3dCrs - is rotated
        relative to the ProjectedCrs, the azimuths remain relative to the
        ProjectedCrs not the local CRS. Note that the projection’s north is
        not the same as true north or magnetic north. A good definition of
        the different kinds of “north” can be found in the OGP Surveying
        &amp; Positioning Guidance Note 1
        http://www.ogp.org.uk/pubs/373-01.pdf (the "True, Grid and Magnetic
        North bearings" paragraph). BUSINESS RULE: Array length equals
        station count
    inclinations
        Dip (or inclination) angle for each station. BUSINESS RULE: Array
        length equals station count
    md_datum
    time_index
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    is_final: bool = field(
        default=None,
        metadata={
            "name": "IsFinal",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    station_count: int = field(
        default=None,
        metadata={
            "name": "StationCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    md_uom: LengthUom = field(
        default=None,
        metadata={
            "name": "MdUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    mds: AbstractDoubleArray = field(
        default=None,
        metadata={
            "name": "Mds",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    first_station_location: Point3d = field(
        default=None,
        metadata={
            "name": "FirstStationLocation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    angle_uom: PlaneAngleUom = field(
        default=None,
        metadata={
            "name": "AngleUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    azimuths: AbstractDoubleArray = field(
        default=None,
        metadata={
            "name": "Azimuths",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    inclinations: AbstractDoubleArray = field(
        default=None,
        metadata={
            "name": "Inclinations",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    md_datum: DataObjectReference = field(
        default=None,
        metadata={
            "name": "MdDatum",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    time_index: Optional[TimeIndex] = field(
        default=None,
        metadata={
            "name": "TimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    witsml_deviation_survey: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "WitsmlDeviationSurvey",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_DoubleTableLookup(AbstractPropertyLookup):
    """Defines a function for table lookups.

    For example, used for linear interpolation, such as PVT. Used for
    categorical property, which also may use StringTableLookup.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: List[DoubleLookup] = field(
        default_factory=list,
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_EarthModelInterpretation(AbstractFeatureInterpretation):
    """An earth model interpretation has a specific role: it gathers a maximum of one of each of these other organization interpretations: structural organization interpretation, stratigraphic organization interpretation, and/or fluid organization interpretation.
    BUSINESS RULE: An earth model Interpretation interprets only an earth model feature.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    stratigraphic_occurrences: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "StratigraphicOccurrences",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    stratigraphic_column: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "StratigraphicColumn",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    structure: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "Structure",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    fluid: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "Fluid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GenericFeatureInterpretation(AbstractFeatureInterpretation):
    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GeologicUnitInterpretation(AbstractFeatureInterpretation):
    """
    The main class for data describing an opinion of a volume-based geologic
    feature or unit.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    geologic_unit_composition: Optional[GeologicUnitComposition] = field(
        default=None,
        metadata={
            "name": "GeologicUnitComposition",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geologic_unit_material_implacement: Optional[
        GeologicUnitMaterialImplacement
    ] = field(
        default=None,
        metadata={
            "name": "GeologicUnitMaterialImplacement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GridConnectionSetRepresentation(AbstractRepresentation):
    """Representation which consists of a list of connections between grid cells,
    potentially on different grids.

    Connections are in the form of (Grid,Cell,Face)1&lt;=&gt;(Grid,Cell,Face)2 and are stored as three integer pair arrays corresponding to these six elements.
    Grid connection sets are the preferred means of representing faults on a grid. The use of cell-face-pairs is more complete than single cell-faces, which are missing a corresponding cell face entry, and only provide an incomplete representation of the topology of a fault.
    Unlike what is sometimes the case in reservoir simulation software, RESQML does not distinguish between standard and non-standard connections.
    Within RESQML if a grid connection corresponds to a "nearest neighbor" as defined by the cell indices, then it is never additive to the implicit nearest neighbor connection.
    BUSINESS RULE: A single cell-face-pair should not appear within more than a single grid connection set. This rule is designed to simplify the interpretation of properties assigned to multiple grid connection sets, which might otherwise have the same property defined more than once on a single connection, with no clear means of resolving the multiple values.

    Parameters
    ----------
    count
        count of connections. Must be positive.
    cell_index_pairs
        2 x #Connections array of cell indices for (Cell1,Cell2) for each
        connection.
    grid_index_pairs
        2 x #Connections array of grid indices for (Cell1,Cell2) for each
        connection. The grid indices are obtained from the grid index pairs.
        If only a single grid is referenced from the grid index, then this
        array need not be used. BUSINESS RULE: This array should appear if
        more than one grid index pair is referenced.
    local_face_per_cell_index_pairs
        Optional 2 x #Connections array of local face per cell indices for
        (Cell1,Cell2) for each connection. Local face per cell indices are
        used because global face indices need not have been defined. Null
        value = -1. If no face per cell definition occur as part of the grid
        representation, e.g., for a block centered grid, then this array
        need not appear.
    connection_interpretations
    grid
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    count: int = field(
        default=None,
        metadata={
            "name": "Count",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cell_index_pairs: AbstractIntegerArray = field(
        default=None,
        metadata={
            "name": "CellIndexPairs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    grid_index_pairs: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "GridIndexPairs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    local_face_per_cell_index_pairs: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "LocalFacePerCellIndexPairs",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    connection_interpretations: Optional[ConnectionInterpretations] = field(
        default=None,
        metadata={
            "name": "ConnectionInterpretations",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    grid: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Grid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_LocalDepth3dCrs(AbstractLocal3dCrs):
    """Defines a local depth coordinate system, the geometrical origin and location
    is defined by the elements of the base class AbstractLocal3dCRS.

    This CRS uses the units of measure of its projected and vertical
    CRS.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_LocalTime3dCrs(AbstractLocal3dCrs):
    """Defines a local time coordinate system, the geometrical origin and location
    is defined by the elements of the base class AbstractLocal3dCRS.

    This CRS defines the time unit that the time-based geometries that
    refers it will use.

    Parameters
    ----------
    time_uom
        Defines the unit of measure of the third (time) coordinates, for the
        geometries that refers to it.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    time_uom: TimeUom = field(
        default=None,
        metadata={
            "name": "TimeUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PointSetRepresentation(AbstractRepresentation):
    """A representation that consists of one or more node patches.

    Each node patch is an array of XYZ coordinates for the 3D points.
    There is no implied linkage between the multiple patches.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    node_patch: List[NodePatch] = field(
        default_factory=list,
        metadata={
            "name": "NodePatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PointsProperty(AbstractProperty):
    """
    Represents the geometric information that should *not* be used as
    representation geometry, but should be used in another context where the
    location or geometrical vectorial distances are needed.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_of_points: List[PatchOfPoints] = field(
        default_factory=list,
        metadata={
            "name": "PatchOfPoints",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PolylineRepresentation(AbstractRepresentation):
    """A representation made up of a single polyline or "polygonal chain", which
    may be closed or not.

    Definition from Wikipedia (http://en.wikipedia.org/wiki/Piecewise_linear_curve):
    A polygonal chain, polygonal curve, polygonal path, or piecewise linear curve, is a connected series of line segments. More formally, a polygonal chain P is a curve specified by a sequence of points \\scriptstyle(A_1, A_2, \\dots, A_n) called its vertices so that the curve consists of the line segments connecting the consecutive vertices.
    In computer graphics a polygonal chain is called a polyline and is often used to approximate curved paths.
    BUSINESS RULE: To record a polyline the writer software must give the values of the geometry of each node in an order corresponding to the logical series of segments (edges). The geometry of a polyline must be a 1D array of points.
    A simple polygonal chain is one in which only consecutive (or the first and the last) segments intersect and only at their endpoints.
    A closed polygonal chain (isClosed=True) is one in which the first vertex coincides with the last one, or the first and the last vertices are connected by a line segment.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


    is_closed: bool = field(
        default=None,
        metadata={
            "name": "IsClosed",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    node_patch: NodePatch = field(
        default=None,
        metadata={
            "name": "NodePatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    line_role: Optional[LineRole] = field(
        default=None,
        metadata={
            "name": "LineRole",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PolylineSetRepresentation(AbstractRepresentation):
    """A representation made up of a set of polylines or a set of polygonal chains
    (for more information, see PolylineRepresentation).

    For compactness, it is organized by line patch as a unique polyline
    set patch. if allPolylineClosed = True, all the polylines are
    connected between the first and the last point. Its geometry is a 1D
    array of points, corresponding to the concatenation of the points of
    all polyline points.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    line_role: Optional[LineRole] = field(
        default=None,
        metadata={
            "name": "LineRole",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    line_patch: List[PolylineSetPatch] = field(
        default_factory=list,
        metadata={
            "name": "LinePatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_RedefinedGeometryRepresentation(AbstractRepresentation):
    """A representation derived from an existing representation by redefining its
    geometry.

    Example use cases include deformation of the geometry of an object,
    change of coordinate system, and change of time &lt;=&gt; depth.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    patch_of_geometry: List[PatchOfGeometry] = field(
        default_factory=list,
        metadata={
            "name": "PatchOfGeometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    supporting_representation: DataObjectReference = field(
        default=None,
        metadata={
            "name": "SupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_RepresentationSetRepresentation(AbstractRepresentation):
    """The parent class of the framework representations.

    It is used to group together individual representations which may be
    of the same kind to represent a “bag” of representations. If the bag
    is homogeneous, then this may be indicated. These “bags” do not
    imply any geologic consistency. For example, you can define a set of
    wellbore frames, a set of wellbore trajectories, a set of blocked
    wellbores. Because the framework representations inherit from this
    class, they inherit the capability to gather individual
    representations into sealed and non-sealed surface framework
    representations, or sealed volume framework representations.

    Parameters
    ----------
    is_homogeneous
        Indicates that all of the selected representations are of a single
        kind.
    representation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    is_homogeneous: bool = field(
        default=None,
        metadata={
            "name": "IsHomogeneous",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    representation: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Representation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StreamlinesRepresentation(AbstractRepresentation):
    """Representation of streamlines associated with a streamline feature and
    interpretation.

    Use StreamlinesFeature to define the vector field that supports the streamlines, i.e., describes what flux is being traced.
    Use Interpretation to distinguish between shared and differing interpretations.
    Usage Note: When defining streamline geometry, the PatchIndex will not be referenced, and may be set to a value of 0.

    Parameters
    ----------
    line_count
        Number of streamlines.
    streamline_wellbores
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    line_count: int = field(
        default=None,
        metadata={
            "name": "LineCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    streamline_wellbores: Optional[StreamlineWellbores] = field(
        default=None,
        metadata={
            "name": "StreamlineWellbores",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geometry: Optional[StreamlinePolylineSetPatch] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StringTableLookup(AbstractPropertyLookup):
    """Defines an integer-to-string lookup table, for example, stores facies properties, where a facies index is associated with a facies name. .
    Used for categorical properties, which also may use a double table lookup."""

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    value: List[StringLookup] = field(
        default_factory=list,
        metadata={
            "name": "Value",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_WellboreFrameRepresentation(AbstractRepresentation):
    """Representation of a wellbore that is organized along a wellbore trajectory
    by its MD values.

    RESQML uses MD values to associate properties on points and to
    organize association of properties on intervals between MD points.

    Parameters
    ----------
    node_count
        Number of nodes. Must be positive.
    node_md
        MD values for each node. BUSINESS RULE: MD values and UOM must be
        consistent with the trajectory representation.
    witsml_log_reference
        The reference to the equivalent WITSML well log.
    interval_stratigraphi_units
    cell_fluid_phase_units
    trajectory
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    node_count: int = field(
        default=None,
        metadata={
            "name": "NodeCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    node_md: AbstractDoubleArray = field(
        default=None,
        metadata={
            "name": "NodeMd",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    witsml_log_reference: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "WitsmlLogReference",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    interval_stratigraphi_units: Optional[IntervalStratigraphicUnits] = field(
        default=None,
        metadata={
            "name": "IntervalStratigraphiUnits",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    cell_fluid_phase_units: Optional[CellFluidPhaseUnits] = field(
        default=None,
        metadata={
            "name": "CellFluidPhaseUnits",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    trajectory: DataObjectReference = field(
        default=None,
        metadata={
            "name": "Trajectory",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_WellboreInterpretation(AbstractFeatureInterpretation):
    """This class contains the data describing an opinion of a borehole.

    This interpretation is relative to one particular well trajectory.

    Parameters
    ----------
    is_drilled
        Used to indicate that this wellbore has been, or is being, drilled.
        This distinguishes from planned wells. For one wellbore feature we
        may expect to have multiple wellbore interpretations: IsDrilled=TRUE
        for instance will be used for updated drilled trajectories.
        IsDrilled=FALSE for planned trajectories.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    is_drilled: bool = field(
        default=None,
        metadata={
            "name": "IsDrilled",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_WellboreTrajectoryRepresentation(AbstractRepresentation):
    """
    Representation of a wellbore trajectory.

    Parameters
    ----------
    start_md
        Specifies the measured depth  for the start of the wellbore
        trajectory. Range may often be from kickoff to TD, but this is not
        necessary. BUSINESS RULE: Start MD is always less than the Finish
        MD.
    finish_md
        Specifies the ending measured depth of the range for the wellbore
        trajectory. Range may often be from kickoff to TD, but this is not
        necessary. BUSINESS RULE: Start MD is always less than the Finish
        MD.
    md_uom
        The unit of measure of the reference MD.
    md_domain
    witsml_trajectory
        Pointer to the WITSML trajectory that is contained in the referenced
        wellbore. (For information about WITSML well and wellbore
        references, see the definition for RESQML technical feature,
        WellboreFeature).
    geometry
        Explicit geometry is not required for vertical wells
    md_datum
    deviation_survey
    parent_intersection
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    start_md: float = field(
        default=None,
        metadata={
            "name": "StartMd",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    finish_md: float = field(
        default=None,
        metadata={
            "name": "FinishMd",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    md_uom: LengthUom = field(
        default=None,
        metadata={
            "name": "MdUom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    md_domain: Optional[MdDomain] = field(
        default=None,
        metadata={
            "name": "MdDomain",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    witsml_trajectory: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "WitsmlTrajectory",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geometry: Optional[AbstractParametricLineGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    md_datum: DataObjectReference = field(
        default=None,
        metadata={
            "name": "MdDatum",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    deviation_survey: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "DeviationSurvey",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    parent_intersection: Optional[WellboreTrajectoryParentIntersection] = field(
        default=None,
        metadata={
            "name": "ParentIntersection",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Address_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_address: Optional[CI_Address] = field(
        default=None,
        metadata={
            "name": "CI_Address",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Date_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_date: Optional[CI_Date] = field(
        default=None,
        metadata={
            "name": "CI_Date",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_OnlineResource_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_online_resource: Optional[CI_OnlineResource] = field(
        default=None,
        metadata={
            "name": "CI_OnlineResource",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Series_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_series: Optional[CI_Series] = field(
        default=None,
        metadata={
            "name": "CI_Series",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Telephone_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_telephone: Optional[CI_Telephone] = field(
        default=None,
        metadata={
            "name": "CI_Telephone",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_Extent_Type(AbstractObject_Type):
    """
    Information about spatial, vertical, and temporal extent.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    description: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    geographic_element: List[EX_GeographicExtent_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "geographicElement",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    temporal_element: List[EX_TemporalExtent_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "temporalElement",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    vertical_element: List[EX_VerticalExtent_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "verticalElement",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractTimePrimitiveType(AbstractTimeObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    related_time: List[RelatedTimeType] = field(
        default_factory=list,
        metadata={
            "name": "relatedTime",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DefinitionType(DefinitionBaseType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    remarks: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractColumnLayerGridGeometry(AbstractGridGeometry):
    """Description of the geometry of a column layer grid, e.g., parity and pinch,
    together with its supporting topology.

    Column layer grid cell geometry is based upon nodes on coordinate
    lines. Geometry is contained within the representation of a grid.
    Point Geometry is that of the column layer coordinate line nodes.
    Coordinate line nodes for all of the coordinate lines, with NKL
    nodes per line. The numbering of these lines follow the pillar
    numbering if no split coordinate lines are present. The unsplit
    coordinate lines share an indexing with the pillars. The numbering
    of the remaining lines are defined in the
    columnsPerSplitCoordinateLine list-of-lists if split coordinate
    lines are present. Pillar numbering is either 1D or 2D, so for
    unfaulted grids, the node dimensions may follow either a 2D or 3D
    array. Otherwise the nodes will be 2D. In HDF5 nodes are stored as
    separate X, Y, Z, values, so add another dimension (size=3) which is
    fastest in HDF5.

    Parameters
    ----------
    kdirection
    pillar_geometry_is_defined
        Indicator that a pillar has at least one node with a defined cell
        geometry. This is considered grid meta-data. If the indicator does
        not indicate that the pillar geometry is defined, then this over-
        rides any other node geometry specification. Array index follows
        #Pillars and so may be either 2d or 1d.
    pillar_shape
    cell_geometry_is_defined
        Indicator that a cell has a defined geometry. This attribute is grid
        metadata. If the indicator shows that the cell geometry is NOT
        defined, then this attribute overrides any other node geometry
        specification. Array index is 2D/3D.
    node_is_colocated_in_kdirection
        Optional indicator that two adjacent nodes on a coordinate line are
        colocated. This is considered grid meta-data, and is intended to
        over-ride any geometric comparison of node locations. Array index
        follows #CoordinateLines x (NKL-1).
    node_is_colocated_on_kedge
        Optional indicator that two adjacent nodes on the KEDGE of a cell
        are colocated. This is considered grid meta-data, and is intended to
        over-ride any geometric comparison of node locations. Array index
        follows #EdgesPerColumn x NKL for unstructured column layer grids
        and 4 x NI x NJ x NKL for IJK grids.
    subnode_topology
    split_coordinate_lines
    split_nodes
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    kdirection: KDirection = field(
        default=None,
        metadata={
            "name": "KDirection",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillar_geometry_is_defined: AbstractBooleanArray = field(
        default=None,
        metadata={
            "name": "PillarGeometryIsDefined",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillar_shape: PillarShape = field(
        default=None,
        metadata={
            "name": "PillarShape",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cell_geometry_is_defined: Optional[AbstractBooleanArray] = field(
        default=None,
        metadata={
            "name": "CellGeometryIsDefined",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    node_is_colocated_in_kdirection: Optional[AbstractBooleanArray] = field(
        default=None,
        metadata={
            "name": "NodeIsColocatedInKDirection",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    node_is_colocated_on_kedge: Optional[AbstractBooleanArray] = field(
        default=None,
        metadata={
            "name": "NodeIsColocatedOnKEdge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    subnode_topology: Optional[ColumnLayerSubnodeTopology] = field(
        default=None,
        metadata={
            "name": "SubnodeTopology",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    split_coordinate_lines: Optional[ColumnLayerSplitCoordinateLines] = field(
        default=None,
        metadata={
            "name": "SplitCoordinateLines",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    split_nodes: Optional[SplitNodePatch] = field(
        default=None,
        metadata={
            "name": "SplitNodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractColumnLayerGridRepresentation(AbstractGridRepresentation):
    """Abstract class that includes IJK grids and unstructured column layer grids.

    All column layer grids have a layer index K=1,...,NK or K0=0,...,NK-1.
    Cell geometry is characterized by nodes on coordinate lines.

    Parameters
    ----------
    nk
        Number of layers in the grid. Must be positive.
    interval_stratigraphic_units
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    nk: int = field(
        default=None,
        metadata={
            "name": "Nk",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    interval_stratigraphic_units: Optional[IntervalStratigraphicUnits] = field(
        default=None,
        metadata={
            "name": "IntervalStratigraphicUnits",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractSeismicSurveyFeature(AbstractTechnicalFeature):
    """An organization of seismic lines. For the context of RESQML, a seismic
    survey does not refer to any vertical dimension information, but only really at
    shot point locations or common midpoint gathers. The seismic traces, if needed
    by reservoir models, are transferred in an industry standard format such as
    SEGY.

    RESQML supports these basic types of seismic surveys:
    - seismic lattice (organization of the traces for the 3D acquisition and processing phases).
    - seismic line (organization of the traces for the 2D acquisition and processing phases).
    Additionally, these seismic lattices and seismic lines can be aggregated into sets.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractStratigraphicOrganizationInterpretation(
    AbstractOrganizationInterpretation
):
    """The main class that defines the relationships between the stratigraphic
    units and provides the stratigraphic hierarchy of the Earth.

    BUSINESS RULE: A stratigraphic organization must be in a ranked order from a lower rank to an upper rank. For example, it is possible to find previous unit containment relationships between several ranks.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    ordering_criteria: OrderingCriteria = field(
        default=None,
        metadata={
            "name": "OrderingCriteria",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractSurfaceFrameworkRepresentation(obj_RepresentationSetRepresentation):
    """Parent class for a sealed or non-sealed surface framework representation.

    Each one instantiates a representation set representation. The
    difference between the sealed and non-sealed frameworks is that, in
    the non-sealed case, we do not have all of the contact
    representations, or we have all of the contacts but they are not all
    sealed.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    contact_identity: List[ContactIdentity] = field(
        default_factory=list,
        metadata={
            "name": "ContactIdentity",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractTruncatedColumnLayerGridRepresentation(AbstractGridRepresentation):
    """Abstract class for truncated IJK grids and truncated unstructured column
    layer grids.

    Each column layer grid class must have a defined geometry in which
    cells are truncated and additional split cells are defined.

    Parameters
    ----------
    nk
        Number of layers in the grid. Must be positive.
    truncation_cells
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    nk: int = field(
        default=None,
        metadata={
            "name": "Nk",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    truncation_cells: TruncationCellPatch = field(
        default=None,
        metadata={
            "name": "TruncationCells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AdditionalGridTopology:
    """Additional grid topology and/or patches, if required, for indexable elements
    that otherwise do not have their topology defined within the grid
    representation.

    For example, column edges need to be defined if we wish to have an
    enumeration for the faces of a column layer grid, but not otherwise.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    split_edges: Optional[SplitEdges] = field(
        default=None,
        metadata={
            "name": "SplitEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    split_nodes: Optional[SplitNodePatch] = field(
        default=None,
        metadata={
            "name": "SplitNodes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    split_column_edges: Optional[ColumnLayerSplitColumnEdges] = field(
        default=None,
        metadata={
            "name": "SplitColumnEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    unstructured_column_edges: Optional[UnstructuredColumnEdges] = field(
        default=None,
        metadata={
            "name": "UnstructuredColumnEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    split_faces: Optional[SplitFaces] = field(
        default=None,
        metadata={
            "name": "SplitFaces",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    ij_split_column_edges: Optional[IjSplitColumnEdges] = field(
        default=None,
        metadata={
            "name": "IjSplitColumnEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    unstructured_subnode_topology: Optional[UnstructuredSubnodeTopology] = field(
        default=None,
        metadata={
            "name": "UnstructuredSubnodeTopology",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    column_layer_subnode_topology: Optional[ColumnLayerSubnodeTopology] = field(
        default=None,
        metadata={
            "name": "ColumnLayerSubnodeTopology",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BoundaryFeatureInterpretation(obj_BoundaryFeatureInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DeviationSurveyRepresentation(obj_DeviationSurveyRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DoubleTableLookup(obj_DoubleTableLookup):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EarthModelInterpretation(obj_EarthModelInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GenericFeatureInterpretation(obj_GenericFeatureInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeologicUnitInterpretation(obj_GeologicUnitInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GpGridUnstructuredGridPatch(Patch):
    """Used to specify unstructured cell grid patch(es) within a general purpose
    grid.

    Multiple patches are supported.

    Parameters
    ----------
    unstructured_cell_count
        Number of unstructured cells. Degenerate case (count=0) is allowed
        for GPGrid.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    unstructured_cell_count: int = field(
        metadata={
            "name": "UnstructuredCellCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: Optional[UnstructuredGridGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GridConnectionSetRepresentation(obj_GridConnectionSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LocalDepth3dCrs(obj_LocalDepth3dCrs):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class LocalTime3dCrs(obj_LocalTime3dCrs):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PointSetRepresentation(obj_PointSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PointsProperty(obj_PointsProperty):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PolylineRepresentation(obj_PolylineRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PolylineSetRepresentation(obj_PolylineSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RedefinedGeometryRepresentation(obj_RedefinedGeometryRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RepresentationSetRepresentation(obj_RepresentationSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StreamlinesRepresentation(obj_StreamlinesRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StringTableLookup(obj_StringTableLookup):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreFrameRepresentation(obj_WellboreFrameRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreInterpretation(obj_WellboreInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreTrajectoryRepresentation(obj_WellboreTrajectoryRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_BlockedWellboreRepresentation(obj_WellboreFrameRepresentation):
    """
    The information that allows you to locate, on one or several grids (existing or
    planned), the intersection of volume (cells) and surface (faces) elements with
    a wellbore trajectory (existing or planned).

    Parameters
    ----------
    cell_count
        The number of non-null entries in the grid indices array.
    cell_indices
        The grid cell index for each blocked well cell. BUSINESS RULE: Array
        length must equal cell count.
    grid_indices
        Size of array = IntervalCount. Null values of -1 signify that that
        interval is not within a grid. BUSINESS RULE: The cell count must
        equal the number of non-null entries in this array.
    local_face_pair_per_cell_indices
        For each cell, these are the entry and exit faces of the trajectory.
        Use null (-1), for instance, at TD when there only one intersection.
        The local face-per-cell index is used because a global face index
        need not have been defined on the grid. BUSINESS RULE: The array
        dimensions must equal 2 x CellCount.
    grid
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_count: int = field(
        default=None,
        metadata={
            "name": "CellCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    cell_indices: AbstractIntegerArray = field(
        default=None,
        metadata={
            "name": "CellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    grid_indices: AbstractIntegerArray = field(
        default=None,
        metadata={
            "name": "GridIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    local_face_pair_per_cell_indices: AbstractIntegerArray = field(
        default=None,
        metadata={
            "name": "LocalFacePairPerCellIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    grid: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Grid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_BoundaryFeature(AbstractGeologicFeature):
    """An interface between two geological objects, such as horizons and faults.

    It is a surface object.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_CategoricalProperty(AbstractValuesProperty):
    """Information specific to one categorical property. Contains discrete integer.

    This type of property is associated either as:
    - an internally stored index to a string through a lookup mapping.
    - an internally stored double to another double value through an explicitly provided table.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    lookup: DataObjectReference = field(
        default=None,
        metadata={
            "name": "Lookup",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_CommentProperty(AbstractValuesProperty):
    """Information specific to one comment property.

    Used to capture comments or annotations associated with a given
    element type in a data-object, for example, associating comments on
    the specific location of a well path.

    Parameters
    ----------
    language
        Identify the language (e.g., US English or French) of the string. It
        is recommended that language names conform to ISO 639.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    language: Optional[str] = field(
        default=None,
        metadata={
            "name": "Language",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_ContinuousProperty(AbstractValuesProperty):
    """Most common type of property used for storing rock or fluid attributes; all
    are represented as doubles.

    So that the value range can be known before accessing all values, the min and max values of the range are also stored.
    BUSINESS RULE: It also contains a unit of measure that can be different from the unit of measure of its property type, but it must be convertible into this unit.

    Parameters
    ----------
    minimum_value
        The minimum of the associated property values. BUSINESS RULE: There
        can be only one value per number of elements.
    maximum_value
        The maximum of the associated property values. BUSINESS RULE: There
        can be only one value per number of elements.
    uom
        Unit of measure for the property.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    uom: ResqmlUom = field(
        default=None,
        metadata={
            "name": "UOM",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    minimum_value: List[float] = field(
        default_factory=list,
        metadata={
            "name": "MinimumValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    maximum_value: List[float] = field(
        default_factory=list,
        metadata={
            "name": "MaximumValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_DiscreteProperty(AbstractValuesProperty):
    """Contains discrete integer values; typically used to store any type of index.

    So that the value range can be known before accessing all values, it
    also stores the minimum and maximum value in the range.

    Parameters
    ----------
    minimum_value
        The minimum of the associated property values. BUSINESS RULE: There
        can only be one value per number of elements.
    maximum_value
        The maximum of the associated property values. BUSINESS RULE: There
        can only be one value per number of elements.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    minimum_value: List[int] = field(
        default_factory=list,
        metadata={
            "name": "MinimumValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    maximum_value: List[int] = field(
        default_factory=list,
        metadata={
            "name": "MaximumValue",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_FaultInterpretation(obj_BoundaryFeatureInterpretation):
    """
    A type of boundary feature, this class contains the data describing an opinion
    about the characterization of the fault, which includes the attributes listed
    below.

    Parameters
    ----------
    is_listric
        Indicates if the normal fault is listric or not. BUSINESS RULE: Must
        be present if the fault is normal. Must not be present if the fault
        is not normal.
    maximum_throw
    mean_azimuth
    mean_dip
    throw_interpretation
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    is_listric: Optional[bool] = field(
        default=None,
        metadata={
            "name": "IsListric",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    maximum_throw: Optional[LengthMeasure] = field(
        default=None,
        metadata={
            "name": "MaximumThrow",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    mean_azimuth: Optional[PlaneAngleMeasure] = field(
        default=None,
        metadata={
            "name": "MeanAzimuth",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    mean_dip: Optional[PlaneAngleMeasure] = field(
        default=None,
        metadata={
            "name": "MeanDip",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    throw_interpretation: List[FaultThrow] = field(
        default_factory=list,
        metadata={
            "name": "ThrowInterpretation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_FrontierFeature(AbstractTechnicalFeature):
    """
    Identifies a frontier or boundary in the earth model that is not a geological
    feature but an arbitrary geographic/geometric surface used to delineate the
    boundary of the model.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GeobodyBoundaryInterpretation(obj_BoundaryFeatureInterpretation):
    """
    A type of boundary feature, this class identifies if the boundary is a geobody
    and the type of the boundary.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    boundary_relation: List[BoundaryRelation] = field(
        default_factory=list,
        metadata={
            "name": "BoundaryRelation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GeobodyInterpretation(obj_GeologicUnitInterpretation):
    """
    A type of rock feature, this class identifies if a rock feature is a geobody
    with any qualifications on the interpretation of the geobody.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    geobody3d_shape: Optional[Geobody3dShape] = field(
        default=None,
        metadata={
            "name": "Geobody3dShape",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GeologicUnitFeature(AbstractGeologicFeature):
    """A volume of rock located between one or more boundary features.

    The limiting boundary features should be genetic boundary features
    (i.e. should not be faults).
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_Grid2dRepresentation(AbstractSurfaceRepresentation):
    """Representation based on a 2D grid.

    For definitions of slowest and fastest axes of the array, see
    Grid2dPatch.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    grid2d_patch: Grid2dPatch = field(
        default=None,
        metadata={
            "name": "Grid2dPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_Grid2dSetRepresentation(AbstractSurfaceRepresentation):
    """Set of representations based on a 2D grid.

    Each 2D grid representation corresponds to one patch of the set.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    grid2d_patch: List[Grid2dPatch] = field(
        default_factory=list,
        metadata={
            "name": "Grid2dPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 2,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_HorizonInterpretation(obj_BoundaryFeatureInterpretation):
    """A type of boundary feature, the class specifies if the boundary feature is a
    horizon.

    Maximum Flooding Surface
    - Transgressive Surface ( for erosion or intrusion ?)
    - Sequence Boundary
    - Stratigraphic Limit
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    boundary_relation: List[BoundaryRelation] = field(
        default_factory=list,
        metadata={
            "name": "BoundaryRelation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    sequence_stratigraphy_surface: Optional[SequenceStratigraphySurface] = field(
        default=None,
        metadata={
            "name": "SequenceStratigraphySurface",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_OrganizationFeature(AbstractGeologicFeature):
    """The explicit description of the relationships between geologic features,
    such as rock features (e.g., stratigraphic units, geobodies, phase unit) and
    boundary features (e.g., genetic, tectonic, and fluid boundaries).

    For types of organizations, see OrganizationKind.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    organization_kind: OrganizationKind = field(
        default=None,
        metadata={
            "name": "OrganizationKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_PlaneSetRepresentation(AbstractSurfaceRepresentation):
    """Defines a plane representation, which can be made up of multiple patches.

    Commonly represented features are fluid contacts or frontiers. Common geometries of this representation are titled or horizontal planes.
    BUSINESS RULE: If the plane representation is made up of multiple patches, then you must specify the outer rings for each plane patch.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    planes: List[AbstractPlaneGeometry] = field(
        default_factory=list,
        metadata={
            "name": "Planes",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_RockFluidOrganizationInterpretation(AbstractOrganizationInterpretation):
    """
    Interpretation of the fluid organization units.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    rock_fluid_unit_index: RockFluidUnitInterpretationIndex = field(
        default=None,
        metadata={
            "name": "RockFluidUnitIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_RockFluidUnitInterpretation(obj_GeologicUnitInterpretation):
    """
    A type of rock fluid feature interpretation , this class identifies if a rock
    fluid feature by its phase.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    phase: Optional[Phase] = field(
        default=None,
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_SealedVolumeFrameworkRepresentation(obj_RepresentationSetRepresentation):
    """A strict boundary representation (BREP), which represents the volume region
    by assembling together shells.

    BUSINESS RULE: The sealed structural framework must be part of the same earth model as this sealed volume framework.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    based_on: DataObjectReference = field(
        default=None,
        metadata={
            "name": "BasedOn",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    shells: List[VolumeShell] = field(
        default_factory=list,
        metadata={
            "name": "Shells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )
    regions: List[VolumeRegion] = field(
        default_factory=list,
        metadata={
            "name": "Regions",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StratigraphicUnitInterpretation(obj_GeologicUnitInterpretation):
    """
    Interpretation of a stratigraphic unit which includes the knowledge of the top,
    the bottom, the deposition mode.

    Parameters
    ----------
    deposition_mode
        BUSINESS RULE / The Deposition mode for a Geological Unit MUST be
        conssitent with the Boundary Relations of A Genetic Boundary. If it
        is not the case the Boundary Relation declaration is retained.
    max_thickness
    min_thickness
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    deposition_mode: Optional[DepositionMode] = field(
        default=None,
        metadata={
            "name": "DepositionMode",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    max_thickness: Optional[LengthMeasure] = field(
        default=None,
        metadata={
            "name": "MaxThickness",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    min_thickness: Optional[LengthMeasure] = field(
        default=None,
        metadata={
            "name": "MinThickness",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StreamlinesFeature(AbstractTechnicalFeature):
    """Specification of the vector field upon which the streamlines are based.

    Streamlines are commonly used to trace the flow of phases (water /
    oil / gas / total) based upon their flux at a specified time. They
    may also be used for trace components for compositional simulation,
    e.g., CO2, or temperatures for thermal simulation. The flux
    enumeration provides support for the most usual cases with provision
    for extensions to other fluxes.

    Parameters
    ----------
    flux
        Specification of the streamline flux, drawn from the enumeration.
    other_flux
        Optional specification of the streamline flux, if an extension is
        required beyond the enumeration. BUSINESS RULE: OtherFlux should
        appear if Flux has the value of other.
    time_index
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    flux: StreamlineFlux = field(
        default=None,
        metadata={
            "name": "Flux",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    other_flux: Optional[str] = field(
        default=None,
        metadata={
            "name": "OtherFlux",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    time_index: TimeIndex = field(
        default=None,
        metadata={
            "name": "TimeIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StructuralOrganizationInterpretation(AbstractOrganizationInterpretation):
    """
    One of the main types of RESQML organizations, this class gathers boundary
    interpretations (e.g., horizons and faults) plus frontier features and their
    relationships (contacts interpretations), which when taken together define the
    structure of a part of the earth.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    ordering_criteria: OrderingCriteria = field(
        default=None,
        metadata={
            "name": "OrderingCriteria",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    faults: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Faults",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    horizons: List[HorizonInterpretationIndex] = field(
        default_factory=list,
        metadata={
            "name": "Horizons",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    sides: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "Sides",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    top_frontier: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "TopFrontier",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    bottom_frontier: List[DataObjectReference] = field(
        default_factory=list,
        metadata={
            "name": "BottomFrontier",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_TriangulatedSetRepresentation(AbstractSurfaceRepresentation):
    """A representation based on set of triangulated mesh patches, which gets its
    geometry from a 1D array of points.

    BUSINESS RULE: The orientation of all the triangles of this representation must be consistent.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    triangle_patch: List[TrianglePatch] = field(
        default_factory=list,
        metadata={
            "name": "TrianglePatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_UnstructuredGridRepresentation(AbstractGridRepresentation):
    """Unstructured grid representation characterized by a cell count, and
    potentially nothing else.

    Both the oldest and newest simulation formats are based on this
    format.

    Parameters
    ----------
    cell_count
        Number of cells in the grid. Must be positive.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    cell_count: int = field(
        default=None,
        metadata={
            "name": "CellCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: Optional[UnstructuredGridGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_WellboreFeature(AbstractTechnicalFeature):
    """May refer to one of these:
    wellbore. A unique, oriented path from the bottom of a drilled borehole to the surface of the earth. The path must not overlap or cross itself.
    borehole. A hole excavated in the earth as a result of drilling or boring operations. The borehole may represent the hole of an entire wellbore (when no sidetracks are present), or a sidetrack extension. A borehole extends from an originating point (the surface location for the initial borehole or kickoff point for sidetracks) to a terminating (bottomhole) point.
    sidetrack. A borehole that originates in another borehole as opposed to originating at the surface.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    witsml_wellbore: Optional[WitsmlWellboreReference] = field(
        default=None,
        metadata={
            "name": "WitsmlWellbore",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_WellboreMarkerFrameRepresentation(obj_WellboreFrameRepresentation):
    """
    A well log frame where each entry represents a well marker.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    wellbore_marker: List[WellboreMarker] = field(
        default_factory=list,
        metadata={
            "name": "WellboreMarker",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Contact_Type(AbstractObject_Type):
    """
    Information required enabling contact with the  responsible person and/or
    organisation.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    phone: Optional[CI_Telephone_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    address: Optional[CI_Address_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    online_resource: Optional[CI_OnlineResource_PropertyType] = field(
        default=None,
        metadata={
            "name": "onlineResource",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    hours_of_service: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "hoursOfService",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    contact_instructions: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "contactInstructions",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EX_Extent(EX_Extent_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Definition(DefinitionType):
    """The basic gml:Definition element specifies a definition, which can be
    included in or referenced by a dictionary.

    The content model for a generic definition is a derivation from
    gml:AbstractGMLType. The gml:description property element shall hold
    the definition if this can be captured in a simple text string, or
    the gml:descriptionReference property element may carry a link to a
    description elsewhere. The gml:identifier element shall provide one
    identifier identifying this definition. The identifier shall be
    unique within the dictionaries using this definition. The gml:name
    elements shall provide zero or more terms and synonyms for which
    this is the definition. The gml:remarks element shall be used to
    hold additional textual information that is not conceptually part of
    the definition but is useful in understanding the definition.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IdentifiedObjectType(DefinitionType):
    """Gml:IdentifiedObjectType provides identification properties of a CRS-related
    object.

    In gml:DefinitionType, the gml:identifier element shall be the
    primary name by which this object is identified, encoding the "name"
    attribute in the UML model. Zero or more of the gml:name elements
    can be an unordered set of "identifiers", encoding the "identifier"
    attribute in the UML model. Each of these gml:name elements can
    reference elsewhere the object's defining information or be an
    identifier by which this object can be referenced. Zero or more
    other gml:name elements can be an unordered set of "alias"
    alternative names by which this CRS related object is identified,
    encoding the "alias" attributes in the UML model. An object may have
    several aliases, typically used in different contexts. The context
    for an alias is indicated by the value of its (optional) codeSpace
    attribute. Any needed version information shall be included in the
    codeSpace attribute of a gml:identifier and gml:name elements. In
    this use, the gml:remarks element in the gml:DefinitionType shall
    contain comments on or information about this object, including data
    source information.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BlockedWellboreRepresentation(obj_BlockedWellboreRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class BoundaryFeature(obj_BoundaryFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CategoricalProperty(obj_CategoricalProperty):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CommentProperty(obj_CommentProperty):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ContinuousProperty(obj_ContinuousProperty):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DiscreteProperty(obj_DiscreteProperty):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FaultInterpretation(obj_FaultInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FrontierFeature(obj_FrontierFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeobodyBoundaryInterpretation(obj_GeobodyBoundaryInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeobodyInterpretation(obj_GeobodyInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeologicUnitFeature(obj_GeologicUnitFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Grid2dRepresentation(obj_Grid2dRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Grid2dSetRepresentation(obj_Grid2dSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class HorizonInterpretation(obj_HorizonInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IjkGridGeometry(AbstractColumnLayerGridGeometry):
    """Explicit geometry definition for the cells of the IJK grid.

    Grid options are also defined through this object.

    Parameters
    ----------
    grid_is_righthanded
        Indicates that the IJK grid is right handed, as determined by the
        triple product of tangent vectors in the I, J, and K directions.
    ij_gaps
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    grid_is_righthanded: bool = field(
        default=None,
        metadata={
            "name": "GridIsRighthanded",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    ij_gaps: Optional[IjGaps] = field(
        default=None,
        metadata={
            "name": "IjGaps",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class OrganizationFeature(obj_OrganizationFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PlaneSetRepresentation(obj_PlaneSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RepresentationIdentity:
    """Indicates the nature of the relationship between 2 or more representations,
    specifically if they are partially or totally identical.

    For possible types of relationships, see IdentityKind.

    Parameters
    ----------
    identical_element_count
        Number of elements within each representation for which a
        representation identity is specified.
    element_identity
    additional_grid_topology
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    identical_element_count: int = field(
        metadata={
            "name": "IdenticalElementCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    element_identity: List[ElementIdentity] = field(
        default_factory=list,
        metadata={
            "name": "ElementIdentity",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 2,
        },
    )
    additional_grid_topology: Optional[AdditionalGridTopology] = field(
        default=None,
        metadata={
            "name": "AdditionalGridTopology",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RockFluidOrganizationInterpretation(obj_RockFluidOrganizationInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RockFluidUnitInterpretation(obj_RockFluidUnitInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SealedVolumeFrameworkRepresentation(obj_SealedVolumeFrameworkRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SeismicLatticeSetFeature(AbstractSeismicSurveyFeature):
    """An unordered set of several seismic lattices.

    Generally, it has no direct interpretation or representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StratigraphicUnitInterpretation(obj_StratigraphicUnitInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StreamlinesFeature(obj_StreamlinesFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StructuralOrganizationInterpretation(obj_StructuralOrganizationInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TriangulatedSetRepresentation(obj_TriangulatedSetRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredColumnLayerGridGeometry(AbstractColumnLayerGridGeometry):
    """Description of the geometry of an unstructured column layer grid, e.g.,
    parity and pinch, together with its supporting topology.

    Unstructured column layer cell geometry is derived from column layer
    cell geometry and hence is based upon nodes on coordinate lines.
    Geometry is contained within the representation of a grid.

    Parameters
    ----------
    column_shape
    pillar_count
        Number of pillars in the grid. Must be positive. Pillars are used to
        describe the shape of the columns in the grid.
    pillars_per_column
        List of pillars for each column. The pillars define the corners of
        each column. The number of pillars per column can be obtained from
        the offsets in the first list of list array. BUSINESS RULE: The
        length of the first array in the list of list construction should
        equal the columnCount.
    column_is_right_handed
        List of columns which are right handed. Right handedness is
        evaluated following the pillar order and the K-direction tangent
        vector for each column.
    column_edges
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    column_shape: ColumnShape = field(
        default=None,
        metadata={
            "name": "ColumnShape",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillar_count: int = field(
        default=None,
        metadata={
            "name": "PillarCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    pillars_per_column: ResqmlJaggedArray = field(
        default=None,
        metadata={
            "name": "PillarsPerColumn",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    column_is_right_handed: AbstractBooleanArray = field(
        default=None,
        metadata={
            "name": "ColumnIsRightHanded",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    column_edges: Optional[UnstructuredColumnEdges] = field(
        default=None,
        metadata={
            "name": "ColumnEdges",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredGridRepresentation(obj_UnstructuredGridRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreFeature(obj_WellboreFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class WellboreMarkerFrameRepresentation(obj_WellboreMarkerFrameRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_CategoricalPropertySeries(obj_CategoricalProperty):
    """Information specific to one comment property.

    Used to capture comments or annotations associated with a given
    element type in a data-object, for example, associating comments on
    the specific location of a well path.

    Parameters
    ----------
    realization_indices
        Provide the list of indices corresponding to realizations number.
        For example, if a user wants to send the realization corresponding
        to p10, p20, ... he would write the array 10, 20, ... If not
        provided, then the realization count (which could be 1) does not
        introduce a dimension to the multi-dimensional array storage.
    series_time_indices
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    realization_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "RealizationIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    series_time_indices: Optional[TimeIndices] = field(
        default=None,
        metadata={
            "name": "SeriesTimeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_CommentPropertySeries(obj_CommentProperty):
    """Information specific to one comment property.

    Used to capture comments or annotations associated with a given
    element type in a data-object, for example, associating comments on
    the specific location of a well path.

    Parameters
    ----------
    realization_indices
        Provide the list of indices corresponding to realizations number.
        For example, if a user wants to send the realization corresponding
        to p10, p20, ... he would write the array 10, 20, ... If not
        provided, then the realization count (which could be 1) does not
        introduce a dimension to the multi-dimensional array storage.
    series_time_indices
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    realization_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "RealizationIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    series_time_indices: Optional[TimeIndices] = field(
        default=None,
        metadata={
            "name": "SeriesTimeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_ContinuousPropertySeries(obj_ContinuousProperty):
    """Information specific to one comment property.

    Used to capture comments or annotations associated with a given
    element type in a data-object, for example, associating comments on
    the specific location of a well path.

    Parameters
    ----------
    realization_indices
        Provide the list of indices corresponding to realizations number.
        For example, if a user wants to send the realization corresponding
        to p10, p20, ... he would write the array 10, 20, ... If not
        provided, then the realization count (which could be 1) does not
        introduce a dimension to the multi-dimensional array storage.
    series_time_indices
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    realization_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "RealizationIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    series_time_indices: Optional[TimeIndices] = field(
        default=None,
        metadata={
            "name": "SeriesTimeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_DiscretePropertySeries(obj_DiscreteProperty):
    """Information specific to one comment property.

    Used to capture comments or annotations associated with a given
    element type in a data-object, for example, associating comments on
    the specific location of a well path.

    Parameters
    ----------
    realization_indices
        Provide the list of indices corresponding to realizations number.
        For example, if a user wants to send the realization corresponding
        to p10, p20, ... he would write the array 10, 20, ... If not
        provided, then the realization count (which could be 1) does not
        introduce a dimension to the multi-dimensional array storage.
    series_time_indices
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    realization_indices: Optional[AbstractIntegerArray] = field(
        default=None,
        metadata={
            "name": "RealizationIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    series_time_indices: Optional[TimeIndices] = field(
        default=None,
        metadata={
            "name": "SeriesTimeIndices",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_FluidBoundaryFeature(obj_BoundaryFeature):
    """A boundary (usually a plane) separating two fluid phases, such as a gas-oil
    contact (GOC), a water-oil contact (WOC), a gas-oil contact (GOC), or others.

    For types, see FluidContact.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    fluid_contact: FluidContact = field(
        default=None,
        metadata={
            "name": "FluidContact",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GeneticBoundaryFeature(obj_BoundaryFeature):
    """A boundary between two units produced by a contrast between two deposits
    that occurred at two different geologic time periods.

    For types, see GeneticBoundaryKind.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    genetic_boundary_kind: GeneticBoundaryKind = field(
        default=None,
        metadata={
            "name": "GeneticBoundaryKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    absolute_age: Optional[Timestamp] = field(
        default=None,
        metadata={
            "name": "AbsoluteAge",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GeobodyFeature(obj_GeologicUnitFeature):
    """A volume of rock that is identified based on some specific attribute, like
    its mineral content or other physical characteristic.

    Unlike stratigraphic or phase units, there is no associated time or
    fluid content semantic. For types, see GeobodyKind.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_NonSealedSurfaceFrameworkRepresentation(
    AbstractSurfaceFrameworkRepresentation
):
    """A collection of contact representations parts, which are a list of contact
    patches with no identity.

    This collection of contact representations is completed by a set of
    representations gathered at the representation set representation
    level.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    non_sealed_contact_representation: List[AbstractContactRepresentationPart] = field(
        default_factory=list,
        metadata={
            "name": "NonSealedContactRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_RockFluidUnitFeature(obj_GeologicUnitFeature):
    """A fluid phase plus one or more stratigraphic units.

    A unit may correspond to a pair of horizons that are not adjacent
    stratigraphically, e.g., a coarse zonation, and is often used to
    define the reservoir. For types, see Phase.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    phase: Phase = field(
        default=None,
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    fluid_boundary_bottom: DataObjectReference = field(
        default=None,
        metadata={
            "name": "FluidBoundaryBottom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    fluid_boundary_top: DataObjectReference = field(
        default=None,
        metadata={
            "name": "FluidBoundaryTop",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_SealedSurfaceFrameworkRepresentation(AbstractSurfaceFrameworkRepresentation):
    """A collection of contact representations parts, which are a list of contact
    patches and their identities.

    This collection of contact representations is completed by a set of
    representations gathered at the representation set representation
    level.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    sealed_contact_representation: List[SealedContactRepresentationPart] = field(
        default_factory=list,
        metadata={
            "name": "SealedContactRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_SeismicLineFeature(AbstractSeismicSurveyFeature):
    """Defined by one lateral dimension: trace (lateral). Seismic trace of the 3D seismic survey.
    To specify its location, the seismic feature can be associated with the seismic coordinates of the points of a representation.

    Parameters
    ----------
    first_trace_index
        The index of the first trace of the seismic line.
    trace_count
        The count of traces in the seismic line.
    trace_index_increment
        The constant index increment between two consecutive traces.
    is_part_of
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    first_trace_index: int = field(
        default=None,
        metadata={
            "name": "FirstTraceIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    trace_count: int = field(
        default=None,
        metadata={
            "name": "TraceCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    trace_index_increment: int = field(
        default=None,
        metadata={
            "name": "TraceIndexIncrement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    is_part_of: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "IsPartOf",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_SeismicLineSetFeature(AbstractSeismicSurveyFeature):
    """An unordered set of several seismic lines.

    Generally, it has no direct interpretation or representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StratigraphicColumnRankInterpretation(
    AbstractStratigraphicOrganizationInterpretation
):
    """
    A global hierarchy containing an ordered list of stratigraphic unit
    interpretations.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    index: int = field(
        default=None,
        metadata={
            "name": "Index",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    stratigraphic_units: List[StratigraphicUnitInterpretationIndex] = field(
        default_factory=list,
        metadata={
            "name": "StratigraphicUnits",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StratigraphicOccurrenceInterpretation(
    AbstractStratigraphicOrganizationInterpretation
):
    """A local Interpretation—it could be along a well, on a 2D map, or on a 2D
    section or on a part of the global volume of an earth model—of a succession of
    rock feature elements.

    The stratigraphic column rank interpretation composing a stratigraphic occurrence can be ordered by the criteria listed in OrderingCriteria.
    BUSINESS RULE: A representation of a stratigraphic occurrence interpretation can be a wellbore marker or a wellbore frame.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    is_occurrence_of: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "IsOccurrenceOf",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geologic_unit_index: List[GeologicUnitInterpretationIndex] = field(
        default_factory=list,
        metadata={
            "name": "GeologicUnitIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_StratigraphicUnitFeature(obj_GeologicUnitFeature):
    """A stratigraphic unit that can have a well-known (e.g., "Jurassic")
    chronostratigraphic top and chronostratigraphic bottom.

    These chronostratigraphic units have no associated interpretations or representations.
    BUSINESS RULE: The name must reference a well-known chronostratigraphic unit (such as "Jurassic"), for example, from the International Commission on Stratigraphy (http://www.stratigraphy.org).
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    chronostratigraphic_bottom: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "ChronostratigraphicBottom",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    chronostratigraphic_top: Optional[DataObjectReference] = field(
        default=None,
        metadata={
            "name": "ChronostratigraphicTop",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_SubRepresentation(AbstractRepresentation):
    """An ordered list of indexable elements and/or indexable element pairs of an
    existing representation.

    Because the representation concepts of topology, geometry, and
    property values are separate in RESQML, it is now possible to select
    a range of nodes, edges, faces, or volumes (cell) indices from the
    topological support of an existing representation to define a sub-
    representation. A sub-representation may describe a different
    feature interpretation using the same geometry or property as the
    "parent" representation. In this case, the only information
    exchanged is a set of potentially non-consecutive indices of the
    topological support of the representation.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"
    supporting_representation: DataObjectReference = field(
        default=None,
        metadata={
            "name": "SupportingRepresentation",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    additional_grid_topology: Optional[AdditionalGridTopology] = field(
        default=None,
        metadata={
            "name": "AdditionalGridTopology",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )

    sub_representation_patch: List[SubRepresentationPatch] = field(
        default_factory=list,
        metadata={
            "name": "SubRepresentationPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_TectonicBoundaryFeature(obj_BoundaryFeature):
    """A boundary caused by tectonic movement or metamorphism, such as a fault or a
    fracture.

    For types, see TectonicBoundaryKind.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    tectonic_boundary_kind: TectonicBoundaryKind = field(
        default=None,
        metadata={
            "name": "TectonicBoundaryKind",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Contact(CI_Contact_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CoordinateSystemAxisType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    axis_abbrev: axisAbbrev = field(
        default=None,
        metadata={
            "name": "axisAbbrev",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    axis_direction: axisDirection = field(
        default=None,
        metadata={
            "name": "axisDirection",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    minimum_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "minimumValue",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    maximum_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "maximumValue",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    range_meaning: Optional[rangeMeaning] = field(
        default=None,
        metadata={
            "name": "rangeMeaning",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EllipsoidType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    semi_major_axis: float = field(
        default=None,
        metadata={
            "name": "semiMajorAxis",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    second_defining_parameter: secondDefiningParameter_2 = field(
        default=None,
        metadata={
            "name": "secondDefiningParameter",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PrimeMeridianType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    greenwich_longitude: float = field(
        default=None,
        metadata={
            "name": "greenwichLongitude",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class domainOfValidity:
    """
    The gml:domainOfValidity property implements an association role to an
    EX_Extent object as encoded in ISO/TS 19139, either referencing or containing
    the definition of that extent.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"

    ex_extent: Optional[EX_Extent] = field(
        default=None,
        metadata={
            "name": "EX_Extent",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CategoricalPropertySeries(obj_CategoricalPropertySeries):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CommentPropertySeries(obj_CommentPropertySeries):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ContinuousPropertySeries(obj_ContinuousPropertySeries):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class DiscretePropertySeries(obj_DiscretePropertySeries):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class FluidBoundaryFeature(obj_FluidBoundaryFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeneticBoundaryFeature(obj_GeneticBoundaryFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeobodyFeature(obj_GeobodyFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GpGridIjkGridPatch(Patch):
    """Used to specify IJK grid patch(es) within a general purpose grid.

    Multiple patches are supported.

    Parameters
    ----------
    ni
        Count of I indices. Degenerate case (ni=0) is allowed for GPGrid
        representations.
    nj
        Count of J indices. Degenerate case (nj=0) is allowed for GPGrid
        representations.
    radial_grid_is_complete
        TRUE if the grid is periodic in J, i.e., has the topology of a
        complete 360 degree circle. If TRUE, then NJL=NJ. Otherwise,
        NJL=NJ+1
    geometry
    truncation_cells
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    ni: int = field(
        metadata={
            "name": "Ni",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    nj: int = field(
        metadata={
            "name": "Nj",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    radial_grid_is_complete: Optional[bool] = field(
        default=None,
        metadata={
            "name": "RadialGridIsComplete",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geometry: Optional[IjkGridGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    truncation_cells: Optional[TruncationCellPatch] = field(
        default=None,
        metadata={
            "name": "TruncationCells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GpGridUnstructuredColumnLayerGridPatch(Patch):
    """Used to specify unstructured column layer grid patch(es) within a general
    purpose grid.

    Multiple patches are supported.

    Parameters
    ----------
    unstructured_column_count
        Number of unstructured columns. Degenerate case (count=0) is allowed
        for GPGrid.
    geometry
    truncation_cells
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    unstructured_column_count: int = field(
        metadata={
            "name": "UnstructuredColumnCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: Optional[UnstructuredColumnLayerGridGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    truncation_cells: Optional[TruncationCellPatch] = field(
        default=None,
        metadata={
            "name": "TruncationCells",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class NonSealedSurfaceFrameworkRepresentation(
    obj_NonSealedSurfaceFrameworkRepresentation
):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RockFluidUnitFeature(obj_RockFluidUnitFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SealedSurfaceFrameworkRepresentation(obj_SealedSurfaceFrameworkRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SeismicLineFeature(obj_SeismicLineFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SeismicLineSetFeature(obj_SeismicLineSetFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StratigraphicColumnRankInterpretation(obj_StratigraphicColumnRankInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StratigraphicOccurrenceInterpretation(obj_StratigraphicOccurrenceInterpretation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class StratigraphicUnitFeature(obj_StratigraphicUnitFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SubRepresentation(obj_SubRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TectonicBoundaryFeature(obj_TectonicBoundaryFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_IjkGridRepresentation(AbstractColumnLayerGridRepresentation):
    """Grid whose topology is characterized by structured column indices (I,J) and
    a layer index, K.

    Cell geometry is characterized by nodes on coordinate lines, where each column of the model has 4 sides. Geometric degeneracy is permitted.
    IJK grids support the following specific extensions:
    IJK radial grids
    K-Layer gaps
    IJ-Column gaps

    Parameters
    ----------
    ni
        Count of cells in the I-direction in the grid. Must be positive.
        I=1,...,NI, I0=0,...,NI-1.
    nj
        Count of cells in the J-direction in the grid. Must be positive.
        J=1,...,NJ, J0=0,...,NJ-1.
    radial_grid_is_complete
        TRUE if the grid is periodic in J, i.e., has the topology of a
        complete 360 degree circle. If TRUE, then NJL=NJ. Otherwise,
        NJL=NJ+1 May be used to change the grid topology for either a
        cartesian or a radial grid, although radial grid usage is by far the
        more common.
    kgaps
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    ni: int = field(
        default=None,
        metadata={
            "name": "Ni",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    nj: int = field(
        default=None,
        metadata={
            "name": "Nj",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    radial_grid_is_complete: Optional[bool] = field(
        default=None,
        metadata={
            "name": "RadialGridIsComplete",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    kgaps: Optional[KGaps] = field(
        default=None,
        metadata={
            "name": "KGaps",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    geometry: Optional[IjkGridGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_RepresentationIdentitySet(AbstractResqmlDataObject):
    """
    A collection of representation identities.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    representation_identity: List[RepresentationIdentity] = field(
        default_factory=list,
        metadata={
            "name": "RepresentationIdentity",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_SeismicLatticeFeature(AbstractSeismicSurveyFeature):
    """Defined by two lateral ordered dimensions: inline (lateral), crossline (lateral and orthogonal to the inline dimension), which are fixed.
    To specify its location, a seismic feature can be associated with the seismic coordinates of the points of a representation.

    Parameters
    ----------
    crossline_count
        The count of crosslines in the 3D seismic survey.
    crossline_index_increment
        The constant index increment between two consecutive crosslines of
        the 3D seismic survey.
    first_crossline_index
        The index of the first crossline of the 3D seismic survey.
    first_inline_index
        The index of the first inline of the 3D seismic survey.
    inline_count
        The count of inlines in the 3D seismic survey.
    inline_index_increment
        The constant index increment between two consecutive inlines of the
        3D seismic survey.
    is_part_of
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    crossline_count: int = field(
        default=None,
        metadata={
            "name": "CrosslineCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    crossline_index_increment: int = field(
        default=None,
        metadata={
            "name": "CrosslineIndexIncrement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    first_crossline_index: int = field(
        default=None,
        metadata={
            "name": "FirstCrosslineIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    first_inline_index: int = field(
        default=None,
        metadata={
            "name": "FirstInlineIndex",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    inline_count: int = field(
        default=None,
        metadata={
            "name": "InlineCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    inline_index_increment: int = field(
        default=None,
        metadata={
            "name": "InlineIndexIncrement",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    is_part_of: Optional[SeismicLatticeSetFeature] = field(
        default=None,
        metadata={
            "name": "IsPartOf",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_TruncatedIjkGridRepresentation(
    AbstractTruncatedColumnLayerGridRepresentation
):
    """Grid class with an underlying IJK topology, together with a 1D split cell
    list.

    The truncated IJK cells have more than the usual 6 faces. The split
    cells are arbitrary polyhedra, identical to those of an unstructured
    cell grid.

    Parameters
    ----------
    ni
        Count of I-indices in the grid. Must be positive.
    nj
        Count of J-indices in the grid. Must be positive.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    ni: int = field(
        default=None,
        metadata={
            "name": "Ni",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    nj: int = field(
        default=None,
        metadata={
            "name": "Nj",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: IjkGridGeometry = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_TruncatedUnstructuredColumnLayerGridRepresentation(
    AbstractTruncatedColumnLayerGridRepresentation
):
    """Grid class with an underlying unstructured column layer topology, together
    with a 1D split cell list.

    The truncated cells have more than the usual number of faces within
    each column. The split cells are arbitrary polyhedra, identical to
    those of an unstructured cell grid.

    Parameters
    ----------
    column_count
        Number of unstructured columns in the grid. Must be positive.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    column_count: int = field(
        default=None,
        metadata={
            "name": "ColumnCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: UnstructuredColumnLayerGridGeometry = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_UnstructuredColumnLayerGridRepresentation(
    AbstractColumnLayerGridRepresentation
):
    """Grid whose topology is characterized by an unstructured column index and a
    layer index, K.

    Cell geometry is characterized by nodes on coordinate lines, where
    each column of the model may have an arbitrary number of sides.

    Parameters
    ----------
    column_count
        Number of unstructured columns in the grid. Must be positive.
    geometry
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    column_count: int = field(
        default=None,
        metadata={
            "name": "ColumnCount",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    geometry: Optional[UnstructuredColumnLayerGridGeometry] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Contact_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_contact: Optional[CI_Contact] = field(
        default=None,
        metadata={
            "name": "CI_Contact",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractCRSType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    domain_of_validity: List[domainOfValidity] = field(
        default_factory=list,
        metadata={
            "name": "domainOfValidity",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    scope: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "min_occurs": 1,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractDatumType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    domain_of_validity: Optional[domainOfValidity] = field(
        default=None,
        metadata={
            "name": "domainOfValidity",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    scope: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "min_occurs": 1,
        },
    )
    anchor_definition: Optional[anchorDefinition] = field(
        default=None,
        metadata={
            "name": "anchorDefinition",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    realization_epoch: Optional[XmlDate] = field(
        default=None,
        metadata={
            "name": "realizationEpoch",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CoordinateSystemAxis(CoordinateSystemAxisType):
    """
    Gml:CoordinateSystemAxis is a definition of a coordinate system axis.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class Ellipsoid_1(EllipsoidType):
    """A gml:Ellipsoid is a geometric figure that may be used to describe the
    approximate shape of the earth.

    In mathematical terms, it is a surface formed by the rotation of an
    ellipse about its minor axis.
    """

    class Meta:
        name = "Ellipsoid"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PrimeMeridian_1(PrimeMeridianType):
    """A gml:PrimeMeridian defines the origin from which longitude values are
    determined.

    The default value for the prime meridian gml:identifier value is
    "Greenwich".
    """

    class Meta:
        name = "PrimeMeridian"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GpGridColumnLayerGrid:
    """Used to construct a column layer grid patch based upon multiple unstructured
    column layer and IJK grids which share a layering scheme.

    Multiple patches are supported.

    Parameters
    ----------
    nk
        Number of layers. Degenerate case (nk=0) is allowed for GPGrid.
    kgaps
    ijk_grid_patch
    unstructured_column_layer_grid_patch
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    nk: int = field(
        metadata={
            "name": "Nk",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
            "required": True,
        }
    )
    kgaps: Optional[KGaps] = field(
        default=None,
        metadata={
            "name": "KGaps",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    ijk_grid_patch: List[GpGridIjkGridPatch] = field(
        default_factory=list,
        metadata={
            "name": "IjkGridPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    unstructured_column_layer_grid_patch: List[
        GpGridUnstructuredColumnLayerGridPatch
    ] = field(
        default_factory=list,
        metadata={
            "name": "UnstructuredColumnLayerGridPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class IjkGridRepresentation(obj_IjkGridRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class RepresentationIdentitySet(obj_RepresentationIdentitySet):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SeismicLatticeFeature(obj_SeismicLatticeFeature):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TruncatedIjkGridRepresentation(obj_TruncatedIjkGridRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class TruncatedUnstructuredColumnLayerGridRepresentation(
    obj_TruncatedUnstructuredColumnLayerGridRepresentation
):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class UnstructuredColumnLayerGridRepresentation(
    obj_UnstructuredColumnLayerGridRepresentation
):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_ResponsibleParty_Type(AbstractObject_Type):
    """
    Identification of, and means of communication with, person(s) and organisations
    associated with the dataset.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"
    role: CI_RoleCode_PropertyType = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    individual_name: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "individualName",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    organisation_name: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "organisationName",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    position_name: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "positionName",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    contact_info: Optional[CI_Contact_PropertyType] = field(
        default=None,
        metadata={
            "name": "contactInfo",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGeneralDerivedCRSType(AbstractCRSType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    conversion: conversion = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CoordinateSystemAxisPropertyType:
    """
    Gml:CoordinateSystemAxisPropertyType is a property type for association roles
    to a coordinate system axis, either referencing or containing the definition of
    that axis.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    coordinate_system_axis: Optional[CoordinateSystemAxis] = field(
        default=None,
        metadata={
            "name": "CoordinateSystemAxis",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EllipsoidPropertyType:
    """
    Gml:EllipsoidPropertyType is a property type for association roles to an
    ellipsoid, either referencing or containing the definition of that ellipsoid.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    ellipsoid: Optional[Ellipsoid_1] = field(
        default=None,
        metadata={
            "name": "Ellipsoid",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class PrimeMeridianPropertyType:
    """
    Gml:PrimeMeridianPropertyType is a property type for association roles to a
    prime meridian, either referencing or containing the definition of that
    meridian.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    prime_meridian: Optional[PrimeMeridian_1] = field(
        default=None,
        metadata={
            "name": "PrimeMeridian",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalDatumType(AbstractDatumType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class obj_GpGridRepresentation(AbstractGridRepresentation):
    """General purpose (GP) grid representation, which includes and/or extends the
    features from all other grid representations.

    This general purpose representation is included in the schema for
    research and/or advanced modeling purposes, but is not expected to
    be used for routine data transfer.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/resqmlv2"

    column_layer_grid: List[GpGridColumnLayerGrid] = field(
        default_factory=list,
        metadata={
            "name": "ColumnLayerGrid",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )
    unstructured_grid_patch: List[GpGridUnstructuredGridPatch] = field(
        default_factory=list,
        metadata={
            "name": "UnstructuredGridPatch",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/resqmlv2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_ResponsibleParty(CI_ResponsibleParty_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalDatum_1(VerticalDatumType):
    """
    Gml:VerticalDatum is a textual description and/or a set of parameters
    identifying a particular reference level surface used as a zero-height surface,
    including its position with respect to the Earth for any of the height types
    recognized by this International Standard.
    """

    class Meta:
        name = "VerticalDatum"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class axis(CoordinateSystemAxisPropertyType):
    """The gml:axis property is an association role (ordered sequence) to the
    coordinate system axes included in this coordinate system.

    The coordinate values in a coordinate tuple shall be recorded in the
    order in which the coordinate system axes associations are recorded,
    whenever those coordinates use a coordinate reference system that
    uses this coordinate system. The gml:AggregationAttributeGroup
    should be used to specify that the axis objects are ordered.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ellipsoid_2(EllipsoidPropertyType):
    """
    Gml:ellipsoid is an association role to the ellipsoid used by this geodetic
    datum.
    """

    class Meta:
        name = "ellipsoid"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class primeMeridian_2(PrimeMeridianPropertyType):
    """
    Gml:primeMeridian is an association role to the prime meridian used by this
    geodetic datum.
    """

    class Meta:
        name = "primeMeridian"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GpGridRepresentation(obj_GpGridRepresentation):
    class Meta:
        namespace = "http://www.energistics.org/energyml/data/resqmlv2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_ResponsibleParty_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_responsible_party: Optional[CI_ResponsibleParty] = field(
        default=None,
        metadata={
            "name": "CI_ResponsibleParty",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractCoordinateSystemType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    axis: List[axis] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "min_occurs": 1,
        },
    )
    aggregation_type: Optional[AggregationType] = field(
        default=None,
        metadata={
            "name": "aggregationType",
            "type": "Attribute",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeodeticDatumType(AbstractDatumType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    prime_meridian: primeMeridian_2 = field(
        default=None,
        metadata={
            "name": "primeMeridian",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    ellipsoid: ellipsoid_2 = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalDatumPropertyType:
    """
    Gml:VerticalDatumPropertyType is property type for association roles to a
    vertical datum, either referencing or containing the definition of that datum.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    vertical_datum: Optional[VerticalDatum_1] = field(
        default=None,
        metadata={
            "name": "VerticalDatum",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Citation_Type(AbstractObject_Type):
    """
    Standardized resource reference.
    """

    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    title: CharacterString_PropertyType = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    alternate_title: List[CharacterString_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "alternateTitle",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    date: List[CI_Date_PropertyType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "min_occurs": 1,
        },
    )
    edition: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    edition_date: Optional[Date_PropertyType] = field(
        default=None,
        metadata={
            "name": "editionDate",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    identifier: List["MD_Identifier_PropertyType"] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    cited_responsible_party: List[CI_ResponsibleParty_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "citedResponsibleParty",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    presentation_form: List[CI_PresentationFormCode_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "presentationForm",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    series: Optional[CI_Series_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    other_citation_details: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "otherCitationDetails",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    collective_title: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "collectiveTitle",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    isbn: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "ISBN",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    issn: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "ISSN",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CartesianCSType(AbstractCoordinateSystemType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EllipsoidalCSType(AbstractCoordinateSystemType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeodeticDatum_1(GeodeticDatumType):
    """
    Gml:GeodeticDatum is a geodetic datum defines the precise location and
    orientation in 3-dimensional space of a defined ellipsoid (or sphere), or of a
    Cartesian coordinate system centered in this ellipsoid (or sphere).
    """

    class Meta:
        name = "GeodeticDatum"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SphericalCSType(AbstractCoordinateSystemType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCSType(AbstractCoordinateSystemType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class verticalDatum_2(VerticalDatumPropertyType):
    """
    Gml:verticalDatum is an association role to the vertical datum used by this
    CRS.
    """

    class Meta:
        name = "verticalDatum"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Citation(CI_Citation_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CartesianCS_1(CartesianCSType):
    """Gml:CartesianCS is a 1-, 2-, or 3-dimensional coordinate system.

    In the 1-dimensional case, it contains a single straight coordinate
    axis. In the 2- and 3-dimensional cases gives the position of points
    relative to orthogonal straight axes. In the multi-dimensional case,
    all axes shall have the same length unit of measure. A CartesianCS
    shall have one, two, or three gml:axis property elements.
    """

    class Meta:
        name = "CartesianCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EllipsoidalCS_1(EllipsoidalCSType):
    """Gml:EllipsoidalCS is a two- or three-dimensional coordinate system in which
    position is specified by geodetic latitude, geodetic longitude, and (in the
    three-dimensional case) ellipsoidal height.

    An EllipsoidalCS shall have two or three gml:axis property elements;
    the number of associations shall equal the dimension of the CS.
    """

    class Meta:
        name = "EllipsoidalCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeodeticDatumPropertyType:
    """
    Gml:GeodeticDatumPropertyType is a property type for association roles to a
    geodetic datum, either referencing or containing the definition of that datum.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    geodetic_datum: Optional[GeodeticDatum_1] = field(
        default=None,
        metadata={
            "name": "GeodeticDatum",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SphericalCS_1(SphericalCSType):
    """Gml:SphericalCS is a three-dimensional coordinate system with one distance
    measured from the origin and two angular coordinates.

    A SphericalCS shall have three gml:axis property elements.
    """

    class Meta:
        name = "SphericalCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCS_1(VerticalCSType):
    """Gml:VerticalCS is a one-dimensional coordinate system used to record the
    heights or depths of points.

    Such a coordinate system is usually dependent on the Earth's gravity
    field, perhaps loosely as when atmospheric pressure is the basis for
    the vertical coordinate system axis. A VerticalCS shall have one
    gml:axis property element.
    """

    class Meta:
        name = "VerticalCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CI_Citation_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    ci_citation: Optional[CI_Citation] = field(
        default=None,
        metadata={
            "name": "CI_Citation",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CartesianCSPropertyType:
    """
    Gml:CartesianCSPropertyType is a property type for association roles to a
    Cartesian coordinate system, either referencing or containing the definition of
    that coordinate system.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    cartesian_cs: Optional[CartesianCS_1] = field(
        default=None,
        metadata={
            "name": "CartesianCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class EllipsoidalCSPropertyType:
    """
    Gml:EllipsoidalCSPropertyType is a property type for association roles to an
    ellipsoidal coordinate system, either referencing or containing the definition
    of that coordinate system.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    ellipsoidal_cs: Optional[EllipsoidalCS_1] = field(
        default=None,
        metadata={
            "name": "EllipsoidalCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SphericalCSPropertyType:
    """
    Gml:SphericalCSPropertyType is property type for association roles to a
    spherical coordinate system, either referencing or containing the definition of
    that coordinate system.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    spherical_cs: Optional[SphericalCS_1] = field(
        default=None,
        metadata={
            "name": "SphericalCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCSPropertyType:
    """
    Gml:VerticalCSPropertyType is a property type for association roles to a
    vertical coordinate system, either referencing or containing the definition of
    that coordinate system.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    vertical_cs: Optional[VerticalCS_1] = field(
        default=None,
        metadata={
            "name": "VerticalCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class geodeticDatum_2(GeodeticDatumPropertyType):
    """
    Gml:geodeticDatum is an association role to the geodetic datum used by this
    CRS.
    """

    class Meta:
        name = "geodeticDatum"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MD_Identifier_Type(AbstractObject_Type):
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"
    code: CharacterString_PropertyType = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "required": True,
        }
    )
    authority: Optional[CI_Citation_PropertyType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class cartesianCS_2(CartesianCSPropertyType):
    """
    Gml:cartesianCS is an association role to the Cartesian coordinate system used
    by this CRS.
    """

    class Meta:
        name = "cartesianCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ellipsoidalCS_2(EllipsoidalCSPropertyType):
    """
    Gml:ellipsoidalCS is an association role to the ellipsoidal coordinate system
    used by this CRS.
    """

    class Meta:
        name = "ellipsoidalCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class sphericalCS_2(SphericalCSPropertyType):
    """
    Gml:sphericalCS is an association role to the spherical coordinate system used
    by this CRS.
    """

    class Meta:
        name = "sphericalCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class verticalCS_2(VerticalCSPropertyType):
    """
    Gml:verticalCS is an association role to the vertical coordinate system used by
    this CRS.
    """

    class Meta:
        name = "verticalCS"
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MD_Identifier(MD_Identifier_Type):
    class Meta:
        namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeodeticCRSType(AbstractCRSType):
    """
    Gml:GeodeticCRS is a coordinate reference system based on a geodetic datum.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"
    geodetic_datum: geodeticDatum_2 = field(
        default=None,
        metadata={
            "name": "geodeticDatum",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    ellipsoidal_cs: Optional[ellipsoidalCS_2] = field(
        default=None,
        metadata={
            "name": "ellipsoidalCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    cartesian_cs: Optional[cartesianCS_2] = field(
        default=None,
        metadata={
            "name": "cartesianCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    spherical_cs: Optional[sphericalCS_2] = field(
        default=None,
        metadata={
            "name": "sphericalCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )



@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCRSType(AbstractCRSType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    vertical_cs: verticalCS_2 = field(
        default=None,
        metadata={
            "name": "verticalCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    vertical_datum: verticalDatum_2 = field(
        default=None,
        metadata={
            "name": "verticalDatum",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GmlVerticalCrsDefinition(AbstractVerticalCrs):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    gml_vertical_crs_definition: VerticalCRSType = field(
        metadata={
            "name": "GmlVerticalCrsDefinition",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class MD_Identifier_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    md_identifier: Optional[MD_Identifier] = field(
        default=None,
        metadata={
            "name": "MD_Identifier",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeodeticCRS(GeodeticCRSType):
    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class VerticalCRS(VerticalCRSType):
    """Gml:VerticalCRS is a 1D coordinate reference system used for recording
    heights or depths.

    Vertical CRSs make use of the direction of gravity to define the
    concept of height or depth, but the relationship with gravity may
    not be straightforward. By implication, ellipsoidal heights (h)
    cannot be captured in a vertical coordinate reference system.
    Ellipsoidal heights cannot exist independently, but only as an
    inseparable part of a 3D coordinate tuple defined in a geographic 3D
    coordinate reference system.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractDQ_Element_Type(AbstractObject_Type):
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"

    name_of_measure: List[CharacterString_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "nameOfMeasure",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    measure_identification: Optional[MD_Identifier_PropertyType] = field(
        default=None,
        metadata={
            "name": "measureIdentification",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    measure_description: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "measureDescription",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    evaluation_method_type: Optional[DQ_EvaluationMethodTypeCode_PropertyType] = field(
        default=None,
        metadata={
            "name": "evaluationMethodType",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    evaluation_method_description: Optional[CharacterString_PropertyType] = field(
        default=None,
        metadata={
            "name": "evaluationMethodDescription",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    evaluation_procedure: Optional[CI_Citation_PropertyType] = field(
        default=None,
        metadata={
            "name": "evaluationProcedure",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    date_time: List[DateTime_PropertyType] = field(
        default_factory=list,
        metadata={
            "name": "dateTime",
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
        },
    )
    result: List[DQ_Result_PropertyType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.isotc211.org/2005/gmd",
            "min_occurs": 1,
            "max_occurs": 2,
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class CRSPropertyType:
    """
    Gml:CRSPropertyType is a property type for association roles to a CRS abstract
    coordinate reference system, either referencing or containing the definition of
    that CRS.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    vertical_crs: Optional[VerticalCRS] = field(
        default=None,
        metadata={
            "name": "VerticalCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    projected_crs: Optional["ProjectedCRS"] = field(
        default=None,
        metadata={
            "name": "ProjectedCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    geodetic_crs: Optional[GeodeticCRS] = field(
        default=None,
        metadata={
            "name": "GeodeticCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GeodeticCRSPropertyType:
    """
    Gml:GeodeticCRSPropertyType is a property type for association roles to a
    geodetic coordinate reference system, either referencing or containing the
    definition of that reference system.
    """

    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    geodetic_crs: Optional[GeodeticCRS] = field(
        default=None,
        metadata={
            "name": "GeodeticCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "pattern": r"other:\w{2,}",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractDQ_PositionalAccuracy_Type(AbstractDQ_Element_Type):
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gmd"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class baseGeodeticCRS(GeodeticCRSPropertyType):
    """
    Gml:baseGeodeticCRS is an association role to the geodetic coordinate reference
    system used by this projected CRS.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class sourceCRS(CRSPropertyType):
    """
    Gml:sourceCRS is an association role to the source CRS (coordinate reference
    system) of this coordinate operation.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class targetCRS(CRSPropertyType):
    """
    Gml:targetCRS is an association role to the target CRS (coordinate reference
    system) of this coordinate operation.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractCoordinateOperationType(IdentifiedObjectType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    domain_of_validity: Optional[domainOfValidity] = field(
        default=None,
        metadata={
            "name": "domainOfValidity",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    scope: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "min_occurs": 1,
        },
    )
    operation_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "operationVersion",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    coordinate_operation_accuracy: List[coordinateOperationAccuracy] = field(
        default_factory=list,
        metadata={
            "name": "coordinateOperationAccuracy",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    source_crs: Optional[sourceCRS] = field(
        default=None,
        metadata={
            "name": "sourceCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    target_crs: Optional[targetCRS] = field(
        default=None,
        metadata={
            "name": "targetCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ProjectedCRSType(AbstractGeneralDerivedCRSType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    base_geodetic_crs: baseGeodeticCRS = field(
        default=None,
        metadata={
            "name": "baseGeodeticCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )
    cartesian_cs: cartesianCS_2 = field(
        default=None,
        metadata={
            "name": "cartesianCS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class GmlProjectedCrsDefinition(AbstractProjectedCrs):
    """
    This is the Energistics encapsulation of the ProjectedCrs type from GML.
    """

    class Meta:
        target_namespace = "http://www.energistics.org/energyml/data/commonv2"

    gml_projected_crs_definition: ProjectedCRSType = field(
        metadata={
            "name": "GmlProjectedCrsDefinition",
            "type": "Element",
            "namespace": "http://www.energistics.org/energyml/data/commonv2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class AbstractGeneralConversionType(AbstractCoordinateOperationType):
    class Meta:
        target_namespace = "http://www.opengis.net/gml/3.2"

    operation_version: Any = field(
        default=None,
        init=False,
        metadata={
            "type": "Ignore",
        },
    )
    source_crs: Any = field(
        default=None,
        init=False,
        metadata={
            "type": "Ignore",
        },
    )
    target_crs: Any = field(
        default=None,
        init=False,
        metadata={
            "type": "Ignore",
        },
    )
    identifier: identifier = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
            "required": True,
        }
    )


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class ProjectedCRS(ProjectedCRSType):
    """Gml:ProjectedCRS is a 2D coordinate reference system used to approximate the
    shape of the earth on a planar surface, but in such a way that the distortion
    that is inherent to the approximation is carefully controlled and known.

    Distortion correction is commonly applied to calculated bearings and
    distances to produce values that are a close match to actual field
    values.
    """

    class Meta:
        namespace = "http://www.opengis.net/gml/3.2"


@dataclass(**({"slots": True,"kw_only":True} if py310 else {}))
class SC_CRS_PropertyType:
    class Meta:
        target_namespace = "http://www.isotc211.org/2005/gsr"

    vertical_crs: Optional[VerticalCRS] = field(
        default=None,
        metadata={
            "name": "VerticalCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    projected_crs: Optional[ProjectedCRS] = field(
        default=None,
        metadata={
            "name": "ProjectedCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    geodetic_crs: Optional[GeodeticCRS] = field(
        default=None,
        metadata={
            "name": "GeodeticCRS",
            "type": "Element",
            "namespace": "http://www.opengis.net/gml/3.2",
        },
    )
    type_value: str = field(
        init=False,
        default="simple",
        metadata={
            "name": "type",
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    arcrole: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    show: Optional[show_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    actuate: Optional[actuate_value] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        },
    )
    uuidref: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    nil_reason: Optional[Union[str, NilReasonEnumeration_value]] = field(
        default=None,
        metadata={
            "name": "nilReason",
            "type": "Attribute",
            "namespace": "http://www.isotc211.org/2005/gco",
            "pattern": r"other:\w{2,}",
        },
    )
