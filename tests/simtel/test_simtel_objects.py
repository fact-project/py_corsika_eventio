import pytest
from pytest import approx
import numpy as np
from eventio import EventIOFile
from eventio.search_utils import (
    yield_toplevel_of_type,
    yield_n_subobjects,
)

prod2_file = 'tests/resources/gamma_test.simtel.gz'
camorgan_v2_file = 'tests/resources/test_camorganv2.simtel.gz'
prod4b_sst1m_file = 'tests/resources/gamma_20deg_0deg_run102___cta-prod4-sst-1m_desert-2150m-Paranal-sst-1m.simtel.gz'
prod4b_astri_file = 'tests/resources/gamma_20deg_0deg_run103___cta-prod4-sst-astri_desert-2150m-Paranal-sst-astri.simtel.gz'
calib_path = 'tests/resources/calib_events.simtel.gz'


test_files = [
    EventIOFile(path) for path in
    [prod2_file, prod4b_astri_file, prod4b_sst1m_file]
]

expected_adc_samples_event1_tel_id_38 = np.load(
    'tests/resources/gamma_test.simtel_event1_tel_id_38_adc_samples.npy'
)


def yield_n_and_assert(f, eventio_type, n=3):
    at_least_one = False
    for x in yield_n_subobjects(f, eventio_type, n=n):
        at_least_one = True
        # assert that what we yield is not None, but somehow meaningful
        assert x
        yield x
    assert at_least_one


def parse_and_assert_consumption(o, limit=0):
    d = o.parse()
    # assert parse() consumed all data from o
    bytes_not_consumed = o.read()
    assert len(bytes_not_consumed) <= limit
    for byte_ in bytes_not_consumed:
        assert byte_ == 0
    return d


def test_70_3_objects():
    from eventio.simtel.objects import History

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, History, n=3)):
            # make sure a History can be iterated and is never empty
            body_reached = False
            for x in o:
                body_reached = True
            assert body_reached


def test_71_3_objects():
    from eventio.simtel.objects import HistoryCommandLine

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, HistoryCommandLine, n=3)):
            t, s = parse_and_assert_consumption(o, limit=3)
            assert isinstance(s, bytes)


def test_72_3_objects():
    from eventio.simtel.objects import HistoryConfig

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, HistoryConfig, n=3)):
            t, s = parse_and_assert_consumption(o, limit=3)
            assert isinstance(s, bytes)


def test_2000_1_object():
    from eventio.simtel.objects import RunHeader

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, RunHeader, n=1)):
            d = parse_and_assert_consumption(o, limit=2)

            assert d['observer'] == b'bernlohr@lfc371.mpi-hd.mpg.de'
            assert d['target'] == b'Monte Carlo beach'


def test_2001_3_objects():
    from eventio.simtel.objects import MCRunHeader

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, MCRunHeader, n=2)):
            parse_and_assert_consumption(o, limit=0)


def test_2002_v3_3_objects():
    from eventio.simtel.objects import CameraSettings

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, CameraSettings, n=3)):
            camera_data = parse_and_assert_consumption(o, limit=0)

            assert o.header.version == 3

            if i == 0:
                # values from pyhessio for 1st object only
                assert camera_data['telescope_id'] == 1
                assert camera_data['n_pixels'] == 1855
                assert camera_data['focal_length'] == np.float32(28.0)
                assert len(camera_data['pixel_x']) == 1855
                assert len(camera_data['pixel_y']) == 1855
                assert camera_data['pixel_x'][1] == np.float32(0.05)
                assert camera_data['pixel_x'][2] == np.float32(0.025)
                assert np.all(camera_data['pixel_shape'] == -1)
                assert camera_data['n_mirrors'] == 198
                assert camera_data['cam_rot'] == 0.1901187151670456


def test_2002_v5_3_objects():
    from eventio.simtel.objects import CameraSettings

    with EventIOFile(prod4b_astri_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, CameraSettings, n=3)):
            cam_data = parse_and_assert_consumption(o, limit=0)

            assert o.header.version == 5
            assert cam_data['n_pixels'] == 2368  # astri
            assert 'effective_focal_length' in cam_data


def test_2003_3_objects():
    from eventio.simtel.objects import CameraOrganization

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, CameraOrganization, n=3)):
            cam_organ = parse_and_assert_consumption(o, limit=1)
            assert cam_organ['telescope_id'] == i + 1

            for sector in cam_organ['sectors']:
                # sector must never contain a zero, unless it is in the
                # very first element
                assert all(s != 0 for s in sector[1:])
                # print(pixel_id, sector)


def test_2003_v2():
    from eventio.simtel.objects import CameraOrganization

    with EventIOFile(camorgan_v2_file) as f:
        for i, o in enumerate(yield_toplevel_of_type(f, CameraOrganization)):
            if o.header.version != 2:
                continue

            cam_organ = parse_and_assert_consumption(o, limit=1)
            assert cam_organ['telescope_id'] == i + 1

            for sector in cam_organ['sectors']:
                # sector must never contain a zero, unless it is in the
                # very first element
                assert all(s != 0 for s in sector[1:])
                # print(pixel_id, sector)


def test_2004():
    from eventio.simtel.objects import PixelSettings

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, PixelSettings, n=3)):
            assert o.telescope_id == i + 1
            pixelset = parse_and_assert_consumption(o, limit=1)
            assert pixelset['n_pixels'] == 1855

    with EventIOFile('tests/resources/prod4_pixelsettings_v3.gz') as f:
        for o in yield_n_and_assert(f, PixelSettings, n=1):
            assert o.header.version == 3

            pixel_settings = o.parse()
            assert 'sum_offset' in pixel_settings


def test_2005_3_objects():
    from eventio.simtel.objects import DisabledPixels

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, DisabledPixels, n=3)):
            pixel_disable = parse_and_assert_consumption(o, limit=0)

            assert pixel_disable['telescope_id'] == i + 1
            assert pixel_disable['n_trig_disabled'] == 0
            assert pixel_disable['n_HV_disabled'] == 0
            assert len(pixel_disable['trigger_disabled']) == 0
            assert len(pixel_disable['HV_disabled']) == 0

    # file with different n_trig_disabled and n_hv_disabled
    with EventIOFile('tests/resources/n_hv_disabled.simtel.gz') as f:
        for i, o in enumerate(yield_n_and_assert(f, DisabledPixels, n=1)):
            pixel_disable = parse_and_assert_consumption(o, limit=0)

            assert pixel_disable['telescope_id'] == 1
            assert pixel_disable['n_trig_disabled'] != 0
            assert pixel_disable['n_HV_disabled'] == 0
            assert len(pixel_disable['trigger_disabled']) != 0
            assert len(pixel_disable['HV_disabled']) == 0
            assert len(pixel_disable['trigger_disabled']) != len(pixel_disable['HV_disabled'])


def test_2006_3_objects():
    from eventio.simtel.objects import CameraSoftwareSettings

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, CameraSoftwareSettings, n=3)):
            d = parse_and_assert_consumption(o, limit=0)

            assert d['telescope_id'] == i + 1
            assert d['dyn_trig_mode'] == 0
            assert d['dyn_trig_threshold'] == 0
            assert d['dyn_HV_mode'] == 0
            assert d['dyn_HV_threshold'] == 0
            assert d['data_red_mode'] == 0
            assert d['zero_sup_mode'] == 0
            assert d['zero_sup_n_thr'] == 0
            assert len(d['zero_sup_thresholds']) == 0
            assert d['unbiased_scale'] == 0
            assert d['dyn_ped_mode'] == 0
            assert d['dyn_ped_events'] == 0
            assert d['dyn_ped_period'] == 0
            assert d['monitor_cur_period'] == 0
            assert d['report_cur_period'] == 0
            assert d['monitor_HV_period'] == 0
            assert d['report_HV_period'] == 0


def test_2007_3_objects():
    from eventio.simtel.objects import PointingCorrection

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, PointingCorrection, n=3)):
            d = parse_and_assert_consumption(o, limit=0)

            assert d['telescope_id'] == i + 1
            assert d['function_type'] == 0
            assert d['n_param'] == 0
            assert len(d['pointing_param']) == 0


def test_2008_3_objects():
    from eventio.simtel.objects import DriveSettings

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, DriveSettings, n=3)):
            assert o.telescope_id == i + 1
            tracking_info = parse_and_assert_consumption(o, limit=0)

        assert tracking_info['range_low_az'] == 0.0
        assert tracking_info['range_low_alt'] == 0.0
        assert tracking_info['range_high_az'] == approx(2 * np.pi)
        assert tracking_info['range_high_alt'] == approx(2 * np.pi)


def test_2009():
    from eventio.simtel.objects import TriggerInformation

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, TriggerInformation, n=3)):
            data = parse_and_assert_consumption(o, limit=2)
            assert 'cpu_time' in data
            assert 'gps_time' in data
            assert 'teltrg_time_by_type' in data

    with EventIOFile('tests/resources/prod4_pixelsettings_v3.gz') as f:
        for o in yield_n_and_assert(f, TriggerInformation, n=10):
            assert o.header.version == 3

            data = parse_and_assert_consumption(o, limit=3)
            assert 'plane_wavefront_compensation' in data
            comp = data['plane_wavefront_compensation']
            assert comp['az'] == 0
            assert comp['alt'] == np.float32(np.pi / 2)
            assert comp['speed_of_light'] == approx(29.97, abs=0.01)


def test_2100_3_objects():
    from eventio.simtel.objects import TrackingPosition

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, TrackingPosition, n=3)):
            pointing = parse_and_assert_consumption(o, limit=0)

            assert 'azimuth_raw' in pointing
            assert 'altitude_raw' in pointing


def test_2200():
    from eventio.simtel.objects import TelescopeEvent

    assert TelescopeEvent.type_to_telid(3305) == 205
    assert TelescopeEvent.type_to_telid(3205) == 105
    assert TelescopeEvent.type_to_telid(2203) == 3


def test_2010():
    from eventio.simtel.objects import TriggerInformation
    # class under test
    from eventio.simtel.objects import ArrayEvent

    with EventIOFile(prod2_file) as f:
        n_events = 0
        for event in yield_toplevel_of_type(f, ArrayEvent):
            assert isinstance(next(event), TriggerInformation)
            n_events += 1
        assert n_events > 0


def test_2011():
    from eventio.simtel.objects import TelescopeEventHeader

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, TelescopeEventHeader, n=3)):
            parse_and_assert_consumption(o, limit=2)

            # print(d)
        # a few printed examples: only version 1!!
        '''
        {
            'loc_count': 1,
            'glob_count': 408,
            'cpu_time': (1408549473, 35597000),
            'gps_time': (0, 0),
            '_t': 1281, # <-- 0x501
             'trg_source': 1,
            'list_trgsect': array([174, 175, 198, 199], dtype=int16),
            'time_trgsect': array(
                [37.537537, 37.537537, 37.537537, 37.537537],
                dtype=float32
            )
        }
        {
            'loc_count': 1,
            'glob_count': 408,
            'cpu_time': (1408549473, 35597000),
            'gps_time': (0, 0),
            '_t': 1281,
            'trg_source': 1,
            'list_trgsect': array(
                [316, 317, 318, 340, 341, 342], dtype=int16),
            'time_trgsect': array(
                [35.285286, 35.285286, 35.285286,
                35.285286, 35.285286, 35.285286],
                dtype=float32
            )
        }
        {
            'loc_count': 1,
            'glob_count': 409,
            'cpu_time': (1408549476, 147099000),
            'gps_time': (0, 0),
            '_t': 1281,
            'trg_source': 1,
            'list_trgsect': array(
                [ 501,  742,  743,  744,  745,  748,  964,  974,  975,
                  1955, 1956, 1960, 2019, 2023, 2024, 2028, 2083, 2087,
                  2088, 2143, 2429, 2430, 2434, 2493, 2497, 2498, 2502,
                  2557, 2561, 2562, 2617, 2622],
                dtype=int16
            ),
            'time_trgsect': array(
            [25.5, 27.5, 27.5, 28.5, 28. , 26.5, 30.5, 30.5, 32. , 28. , 28.5,
            28.5, 28.5, 27. , 26.5, 27.5, 27.5, 27.5, 27.5, 30.5, 31. , 29. ,
            29. , 29. , 27. , 27. , 27. , 27. , 27. , 27. , 31. , 31. ],
            dtype=float32)
        }
        {
            'loc_count': 1,
            'glob_count': 409,
            'cpu_time': (1408549476, 147099000),
            'gps_time': (0, 0),
            '_t': 1281,
            'trg_source': 1,
            'list_trgsect': array([2569, 2570], dtype=int16),
            'time_trgsect': array([27., 27.], dtype=float32)
        }
        '''

    with EventIOFile('tests/resources/prod4_pixelsettings_v3.gz') as f:
        for i, o in enumerate(yield_n_and_assert(f, TelescopeEventHeader, n=10)):
            o = parse_and_assert_consumption(o, limit=3)
            assert 'relative_trigger_time' in o
            assert 'readout_time' in o


def test_2012_3_objects():
    from eventio.simtel.objects import ADCSums

    with EventIOFile(prod4b_astri_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, ADCSums, n=3)):
            parse_and_assert_consumption(o, limit=3)


def test_2013_3_objects():
    from eventio.simtel.objects import ADCSamples

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, ADCSamples, n=3)):
            d = parse_and_assert_consumption(o, limit=3)

            if i == 0:
                assert o.telescope_id == 38
                assert d.shape == expected_adc_samples_event1_tel_id_38.shape
                assert (d == expected_adc_samples_event1_tel_id_38).all()


def test_2014_3_objects():
    from eventio.simtel.objects import ImageParameters

    expected_telescope_ids = [
        38, 47, 11, 21, 24, 26, 61, 63, 118, 119, 17, 104, 124, 2, 4, 14,
        15, 17, 19, 2, 3, 4, 10, 12, 25, 8, 16, 26, 28, 1, 3, 4, 9, 12,
        25, 62, 110, 126, 9, 12, 22, 25, 27, 62
    ]

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, ImageParameters, n=3)):
            d = parse_and_assert_consumption(o, limit=2)

            assert d['telescope_id'] == expected_telescope_ids[i]


def test_2015_3_objects():
    from eventio.simtel.objects import StereoReconstruction

    with EventIOFile(prod4b_sst1m_file) as f:
        for obj in yield_n_and_assert(f, StereoReconstruction, n=3):
            parse_and_assert_consumption(obj, limit=2)

            # print(d)
            # Does this look reasonable??
            '''
            {
                'result_bits': 5,
                'n_trg': 2,
                'n_read': 2,
                'n_img': 2,
                'img_pattern': 268435472,
                'Az': 6.265805721282959,
                'Alt': 1.2318674325942993,
                'xc': -715.0912475585938,
                'yc': 785.6849365234375
            }
            '''


def test_2016():
    from eventio.simtel.objects import PixelTiming

    with EventIOFile(prod4b_sst1m_file) as f:
        for obj in yield_n_and_assert(f, PixelTiming, n=15):
            timing = parse_and_assert_consumption(obj, limit=3)

            time = timing['time'][:, 0]
            pixels = timing['pixel_list']

            if timing['list_type'] == 1:
                mask = np.isfinite(time)
                assert mask.sum() == len(timing['pixel_list'])
                assert not np.any(np.isnan(time[pixels]))


@pytest.mark.xfail
def test_2017():
    assert False


def test_2020_3_objects():
    from eventio.simtel.objects import MCShower

    hessio_data = np.load('tests/resources/gamma_test_mc_shower.npy')
    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, MCShower, n=3)):
            mc = parse_and_assert_consumption(o, limit=2)

            expected = hessio_data[i]
            assert mc['primary_id'] == 0
            assert mc['energy'] == expected[0]
            assert mc['h_first_int'] == expected[1]
            assert mc['xmax'] == expected[2]


def test_2021_3_objects():
    from eventio.simtel.objects import MCEvent

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, MCEvent, n=3)):
            d = parse_and_assert_consumption(o, limit=0)

            assert d['shower_num'] == o.header.id // 100
            '''
            {
                'event': 11909,
                'shower_num': 119,
                'xcore': 1050.17626953125,
                'ycore': 1743.0797119140625
            }
            '''


def test_2022_3_objects():
    from eventio.simtel.objects import CameraMonitoring

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, CameraMonitoring, n=3)):
            d = parse_and_assert_consumption(o, limit=1)

            assert d['telescope_id'] == i + 1

            # print(d)
            # Looks reasonable
            '''
            {
                'status_time': (1408549473, 35597000),
                'trig_time': (1408549473, 35597000),
                'ped_noise_time': (1408549473, 35597000),
                'hv_temp_time': (1408549473, 35597000),
                'dc_rate_time': (1408549473, 35597000),
                'hv_thr_time': (1408549473, 35597000),
                'set_daq_time': (1408549473, 35597000),
                'status_bits': 4778,
                'coinc_count': 0,
                'event_count': 0,
                'trigger_rate': 750.0,
                'sector_rate': array([34.910618, 26.935232, 34.35181 ,
                ..., 38.751358, 28.185534, 33.873787], dtype=float32),
                'event_rate': 700.0,
                'data_rate': 1.2999999523162842,
                'mean_significant': 0.0,
                'n_ped_slices': 30,
                'pedestal': array([[2979.467 , 3009.359 , 3010.3691,
                ..., 2990.7085, 2929.3687, 2981.3044]], dtype=float32),
                'noise': array([[5.477226, 5.477226, 5.477226,
                ..., 5.477226, 5.477226, 5.477226]], dtype=float32),
                'n_drawer_temp': 0,
                'n_camera_temp': 0,
                'hv_v_mon': array(
                    [836, 814, 823, ..., 893, 858, 847], dtype=int16),
                'hv_i_mon': array(
                    [118, 114, 116, ..., 126, 121, 119], dtype=int16),
                'hv_stat': array([1, 1, 1, ..., 1, 1, 1], dtype=uint8),
                'drawer_temp': array([], shape=(116, 0), dtype=int16),
                'camera_temp': array([], dtype=int16),
                'current': array([18, 19, 19, ..., 18, 18, 19], dtype=uint16),
                'scaler': array(
                    [201, 201, 201,
                    ..., 201, 201, 201], dtype=uint16),
                'hv_dac': array(
                    [ 983,  958,  968,
                    ..., 1051, 1009,  997], dtype=uint16),
                'thresh_dac': array([
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870, 6870,
                   6870, 6870, 6870, 6870, 6870, 6870], dtype=uint16),
                'hv_set': array([1, 1, 1, ..., 1, 1, 1], dtype=uint8),
                'trig_set': array([1, 1, 1, ..., 1, 1, 1], dtype=uint8),
                'daq_conf': 1,
                'daq_scaler_win': 0,
                'daq_nd': 0,
                'daq_acc': 0,
                'daq_nl': 30
            }
            '''


def test_2023():
    from eventio.simtel.objects import LaserCalibration

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, LaserCalibration, n=3)):
            d = parse_and_assert_consumption(o, limit=0)

            assert o.header.version == 2
            assert d['telescope_id'] == i + 1
            assert d['lascal_id'] == 2
            assert 'flat_fielding' not in d

    '''
    {
    'telescope_id': 1,
    'lascal_id': 2,
    'max_int_frac': array([0.], dtype=float32),
    'max_pixtm_frac': array([0.], dtype=float32),
    'calib': array([
        [0.02838226, 0.02617863, 0.02520496,
        ..., 0.02812363, 0.02769747, 0.02691549]], dtype=float32),
    'tm_calib': array([
        [-21.383808, -21.283247, -21.452444,
        ..., -22.023653, -21.650948, -21.601557]],
        dtype=float32)}
    '''

    with EventIOFile('tests/resources/test_lasercal_v3.simtel.gz') as f:
        for i, o in enumerate(yield_n_and_assert(f, LaserCalibration, n=4)):
            d = parse_and_assert_consumption(o, limit=0)

            assert o.header.version == 3
            assert d['telescope_id'] == i + 1
            assert d['lascal_id'] == 2
            assert 'flat_fielding' in d
            assert d['flat_fielding'].shape == d['calib'].shape


@pytest.mark.xfail
def test_2024():
    assert False


@pytest.mark.xfail
def test_2025():
    assert False


def test_2026_3_objects():
    from eventio.simtel.objects import MCPhotoelectronSum

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, MCPhotoelectronSum, n=3)):
            d = parse_and_assert_consumption(o, limit=0)

            assert d['event'] // 100 == d['shower_num']


def test_2027_3_objects():
    from eventio.simtel.objects import PixelList

    with EventIOFile(prod2_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, PixelList, n=3)):
            d = parse_and_assert_consumption(o, limit=2)
            assert d

    # does this look reasonable?
    '''
    {
        'code': 0, 'telescope': 38, 'pixels': 2,
        'pixel_list': array([670, 764], dtype=int16)
    }
    {
        'code': 1, 'telescope': 38, 'pixels': 2,
        'pixel_list': array([671, 718], dtype=int16)
    }
    {
        'code': 0, 'telescope': 47, 'pixels': 3,
        'pixel_list': array([  68, 1242, 1338], dtype=int16)
    }
    '''


def test_2028():
    from eventio.simtel.objects import CalibrationEvent, ArrayEvent

    with EventIOFile(calib_path) as f:
        n_events = 0
        for event in yield_toplevel_of_type(f, CalibrationEvent):
            assert isinstance(next(event), ArrayEvent)
            n_events += 1
        assert n_events > 0


@pytest.mark.xfail
def test_2029():
    assert False


@pytest.mark.xfail
def test_2030():
    assert False


@pytest.mark.xfail
def test_2031():
    assert False


def test_2032():
    from eventio.simtel.objects import PixelTriggerTimes

    with EventIOFile(prod4b_astri_file) as f:
        for i, o in enumerate(yield_n_and_assert(f, PixelTriggerTimes, n=3)):
            d = parse_and_assert_consumption(o, limit=2)
            print(d)
            assert 'n_times' in d
            assert 'pixel_ids' in d
            assert 'trigger_times' in d
