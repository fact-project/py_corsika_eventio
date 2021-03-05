import eventio
import numpy as np
from pytest import approx

testfile = 'tests/resources/one_shower.dat'
prod4_simtel = 'tests/resources/gamma_20deg_0deg_run103___cta-prod4-sst-astri_desert-2150m-Paranal-sst-astri.simtel.gz'

test_emitter_file = 'tests/resources/proton_500GeV_iactext.eventio.gz'
test_profile_file = 'tests/resources/gamma_100gev_1216.eventio'


def test_photo_electrons():
    from eventio import EventIOFile
    from eventio.iact import PhotoElectrons
    from eventio.search_utils import yield_n_subobjects

    with EventIOFile(prod4_simtel) as f:
        for o in yield_n_subobjects(f, PhotoElectrons):
            data = o.parse()

            # astri's number of pixels
            assert data['n_pixels'] == 2368
            assert data['n_pe'] > 0
            assert 0 <= data['non_empty'] <= data['n_pixels']
            assert len(data['photoelectrons']) == data['n_pixels']
            assert data['photoelectrons'].sum() == data['n_pe']
            assert len(data['pixel_id']) == data['n_pe']
            assert len(data['time']) == data['n_pe']

            # times should be within 200 nanoseconds
            assert np.all(0 <= data['time'])
            assert np.all(data['time'] <= 200)

            not_read = o.read()
            assert len(not_read) == 0 or all(b == 0 for b in not_read)


def test_file_has_run_header():
    from eventio.iact import RunHeader
    with eventio.EventIOFile(testfile) as f:
        obj = next(f)
        assert isinstance(obj, RunHeader)


def test_read_run_header():
    with eventio.EventIOFile(testfile) as f:
        run_header = next(f).parse()

        assert run_header['energy_min'] == approx(5.0)
        assert run_header['energy_max'] == approx(100.0)
        assert run_header['energy_spectrum_slope'] == approx(-2.7)


def test_telescope_definition():
    from eventio.iact import TelescopeDefinition

    with eventio.EventIOFile(testfile) as f:
        next(f)  # skip run_header
        next(f)  # skip inputcard

        telescope_definition = next(f)
        assert isinstance(telescope_definition, TelescopeDefinition)
        assert telescope_definition.n_telescopes == 1

        telescope_definition_data = telescope_definition.parse()
        assert telescope_definition_data['x'][0] == approx(0.0)
        assert telescope_definition_data['y'][0] == approx(0.0)
        assert telescope_definition_data['z'][0] == approx(2500.0)
        assert telescope_definition_data['r'][0] == approx(2500.0)


def test_corsika_event_header():
    from eventio.iact import EventHeader
    with eventio.EventIOFile(testfile) as f:
        # first event header should be the 4th object in the file
        for i in range(4):
            obj = next(f)

        assert isinstance(obj, EventHeader)
        event_header = obj.parse()
        assert event_header['event_number'] == 1
        assert event_header['zenith'] == approx(0.0)
        assert event_header['azimuth'] == approx(0.0)
        assert event_header['total_energy'] == approx(9.3249321)


def test_corsika_array_offsets():
    from eventio.iact import ArrayOffsets
    with eventio.EventIOFile(testfile) as f:
        # first array offset object should be the 5th object in the file
        for i in range(5):
            obj = next(f)

        assert isinstance(obj, ArrayOffsets)

        time_offset, offsets = obj.parse()
        assert len(offsets) == 1
        assert offsets['x'][0] == approx(-506.9717102050781)
        assert offsets['y'][0] == approx(-3876.447265625)


def test_event_has_382_bunches():
    from eventio.iact import Photons, TelescopeData
    with eventio.EventIOFile(testfile) as f:
        # first telescope data  object should be the 6th object in the file
        for i in range(6):
            obj = next(f)

        assert isinstance(obj, TelescopeData)
        photons = next(obj)
        assert isinstance(photons, Photons)
        assert photons.n_bunches == 382


def test_bunches():
    from eventio.iact import TelescopeData

    columns = ('x', 'y', 'cx', 'cy', 'time', 'zem', 'photons', 'wavelength')
    with eventio.EventIOFile(testfile) as f:
        # first telescope data  object should be the 6th object in the file
        for i in range(6):
            obj = next(f)

        assert isinstance(obj, TelescopeData)
        photons = next(obj)
        bunches, emitter = photons.parse()

        assert bunches.dtype.names == columns
        assert emitter is None


def test_bunches_chunked():
    from eventio.iact import TelescopeData

    with eventio.EventIOFile(testfile) as f:
        # first telescope data  object should be the 6th object in the file
        for i in range(6):
            obj = next(f)

        assert isinstance(obj, TelescopeData)

        photons = next(obj)
        for i, (bunches, emitter) in enumerate(photons.parse(chunksize=100)):
            if i < 3:
                assert len(bunches) == 100

        assert i == 3


def test_emitter_bunches():
    from eventio.iact import TelescopeData

    columns = ('x', 'y', 'cx', 'cy', 'time', 'zem', 'photons', 'wavelength')
    with eventio.EventIOFile(test_emitter_file) as f:
        # first telescope data  object should be the 6th object in the file
        for i in range(7):
            obj = next(f)

        assert isinstance(obj, TelescopeData)
        photons = next(obj)
        bunches, emitter = photons.parse()

        assert bunches.dtype.names == columns
        assert len(emitter) == len(bunches)
        assert np.all(emitter['wavelength'] == np.float32(9999))

        # lightest particle should be an electron
        assert np.isclose(np.unique(emitter['mass'])[0], 0.000511, rtol=0.1)

        # second lightest particle should be a muon
        assert np.isclose(np.unique(emitter['mass'])[1], 0.105, rtol=0.1)


def test_atmospheric_profile():
    from eventio.iact.objects import AtmosphericProfile

    atmprof8 = np.genfromtxt('tests/resources/atmprof8.dat')

    with eventio.EventIOFile(test_profile_file) as f:
        for i in range(3):
            o = next(f)

        assert isinstance(o, AtmosphericProfile)

        profile = o.parse()

        assert profile['id'] == 8
        assert profile['name'] == b'atmprof8.dat'
        assert np.allclose(profile['altitude_km'], atmprof8[:, 0])
        assert np.allclose(profile['rho'], atmprof8[:, 1])
        assert np.allclose(profile['thickness'], atmprof8[:, 2])
        assert np.allclose(profile['refractive_index_minus_1'], atmprof8[:, 3])
