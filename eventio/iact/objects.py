''' Methods to read in and parse the IACT EventIO object types '''
import struct
import numpy as np
from io import BytesIO
from corsikaio.subblocks import (
    parse_run_header,
    parse_run_end,
    parse_event_header,
    parse_event_end,
)

from ..tools import (
    read_short, read_int, read_float, read_from, read_string,
    read_array, read_var_string, read_double, read_unsigned_varint
)
from ..base import EventIOObject
from ..exceptions import WrongSize
from ..version_handling import assert_version_in, assert_max_version


__all__ = [
    'RunHeader',
    'TelescopeDefinition',
    'EventHeader',
    'ArrayOffsets',
    'TelescopeData',
    'Photons',
    'CameraLayout',
    'TriggerTime',
    'PhotoElectrons',
    'EventEnd',
    'RunEnd',
    'Longitudinal',
    'InputCard',
]


class RunHeader(EventIOObject):
    '''
    This object contains the corsika run header block
    '''
    eventio_type = 1200

    def parse(self):
        '''
        Read the data in this EventIOItem

        Returns a dictionary with the items of the  run header block
        '''
        self.seek(0)
        stream = BytesIO(self.read())
        n = read_int(stream)
        if n != 273:
            raise WrongSize('Expected 273 floats, but found {}'.format(n))

        return parse_run_header(stream.read())[0]


class TelescopeDefinition(EventIOObject):
    '''
    This object contains the coordinates of the telescopes
    of the simulated array
    '''
    eventio_type = 1201

    def __init__(self, header, filehandle):
        super().__init__(header, filehandle)
        self.n_telescopes = read_int(self)

    def __len__(self):
        return self.n_telescopes

    def parse(self):
        '''
        Read the data in this EventIOItem

        Returns a structured numpy array with columns (x, y, z, r)
        with a row for each telescope
        '''
        self.seek(4)
        data = self.read()

        number_of_following_arrays = len(data) // (self.n_telescopes * 4)
        if number_of_following_arrays != 4:
            # DN: I think this cannot happen, but who knows.
            msg = 'Number_of_following_arrays is: {}'
            raise Exception(msg.format(number_of_following_arrays))

        dtype = np.dtype('float32')
        block = np.frombuffer(
            data,
            dtype=dtype,
            count=self.n_telescopes * dtype.itemsize,
        )
        tel_pos = np.core.records.fromarrays(
            block.reshape(4, self.n_telescopes),
            names=['x', 'y', 'z', 'r'],
        )

        return tel_pos


class EventHeader(EventIOObject):
    ''' This Object contains the  event header block '''
    eventio_type = 1202

    def parse(self):
        '''
        Read the data in this EventIOItem

        Returns a dictionary containing the keys of the
         event header block
        '''
        self.seek(0)
        data = self.read()
        n, = struct.unpack('i', data[:4])
        if n != 273:
            raise WrongSize(
                'Expected 273 floats, but found {}'.format(n))

        return parse_event_header(data[4:])[0]

    def __str__(self):
        return super().__str__() + '(event_id={})'.format(self.header.id)


class ArrayOffsets(EventIOObject):
    eventio_type = 1203
    dtypes = {
        0: [('x', 'f4'), ('y', 'f4')],
        1: [('x', 'f4'), ('y', 'f4'), ('weight', 'f4')],
    }

    def parse(self):
        '''
        Read the data in this EventIOItem

        Returns a structured numpy array with columns (t, x, y, weight)
        with a row for each array. This object is used to store the
        array position and contains one set of coordinates for each reuse.
        '''
        assert_max_version(self, 1)
        self.seek(0)

        n_arrays = read_int(self)
        time_offset = read_float(self)

        return time_offset, read_array(
            self,
            dtype=self.dtypes[self.header.version],
            count=n_arrays,
        )


class TelescopeData(EventIOObject):
    '''
    A container class for the photon bunches.
    Usually contains one photon bunch object (Photons)
    per simulated telescope
    '''
    eventio_type = 1204

    def __str__(self):
        return '{}[{}](event_id={})'.format(
            self.__class__.__name__,
            self.header.type,
            self.header.id,
        )


class Photons(EventIOObject):
    '''
    This object contains the data of the simulated cherenkov photons
    for a single telescope
    '''
    eventio_type = 1205
    columns = ('x', 'y', 'cx', 'cy', 'time', 'zem', 'photons', 'wavelength')
    particle_columns = ('x', 'y', 'cx', 'cy', 'time', 'momentum', 'weight', 'particle_id')
    emitter_columns = (
        'x', 'y', 'mass', 'charge', 'time', 'emission_time', 'energy', 'wavelength'
    )

    compact_dtype = np.dtype([(c, 'int16') for c in columns])
    long_dtype = np.dtype([(c, 'float32') for c in columns])
    particle_dtype = np.dtype([(c, 'float32') for c in particle_columns])
    emitter_dtype = np.dtype([(c, 'float32') for c in emitter_columns])

    def __init__(self, header, filehandle):
        super().__init__(header, filehandle)
        self.compact = bool(self.header.version // 1000 == 1)
        self.array_id = self.header.id // 1000
        self.telescope_id = self.header.id % 1000

        (
            self.array,
            self.telescope,
            self.n_photons,
            self.n_bunches
        ) = read_from(self, 'hhfi')

    def __str__(self):
        # IACTEXT writes particles at obslevel into photon bunch
        # objects with ids set to 999
        if self.array_id == 999 and self.telescope_id == 999:
            return 'ObservationLevelParticles(n_particles={})'.format(self.n_bunches)

        return '{}[{}](array_id={}, telescope_id={}, n_bunches={})'.format(
            self.__class__.__name__,
            self.eventio_type,
            self.array_id,
            self.telescope_id,
            self.n_bunches,
        )

    def parse(self, chunksize=None):
        '''
        Read the data in this EventIOObject

        Returns a numpy structured array with a record for each photon
        and the following columns:
            x:         x coordinate in cm
            y:         y coordinate in cm
            cx:        cosine of incident angle in x direction
            cy:        cosine of incident angle in y direction
            time:      time since first interaction in ns
            zem:       Emission height in cm above sea level
            wavelength:    wavelength in nm
            scattered: indicates if the photon was scattered in the atmosphere
        '''
        self.seek(12)

        if (self.array_id == 999 and self.telescope_id == 999):
            data = self.read_data()
            return data.view(self.particle_dtype)

        if chunksize is None:
            data = self.read_data()
            return self.parse_emitter(data)
        else:
            return self.parse_chunked(chunksize)

    def parse_chunked(self, chunksize):
        n_chunks = int(np.ceil(self.n_bunches / chunksize))
        last_chunksize = self.n_bunches % chunksize

        # read the first full size chunks
        for i in range(n_chunks - 1):
            data = self.read_data(chunksize=chunksize)
            yield self.parse_emitter(data)

        # read the remaining, smaller chunk
        data = self.read_data(chunksize=last_chunksize)
        yield self.parse_emitter(data)

    def read_data(self, chunksize=None):
        if self.compact:
            dtype = self.compact_dtype
        else:
            dtype = self.long_dtype

        if self.n_bunches == 0:
            return np.array([], dtype=dtype)

        count = chunksize or self.n_bunches

        bunches = np.frombuffer(
            self.read(count * dtype.itemsize),
            dtype=dtype,
            count=count
        )

        if self.compact:
            bunches = self.parse_compact(bunches)

        return bunches

    def parse_emitter(self, data):
        emitter_mask = data['wavelength'] == np.float32(9999)
        if np.any(emitter_mask):
            photons = data[~emitter_mask]
            emitter = data[emitter_mask].view(self.emitter_dtype)
        else:
            photons = data
            emitter = None

        return photons, emitter

    @classmethod
    def parse_compact(cls, data):
        data = data.astype(cls.long_dtype)
        data['x'] *= 0.1  # now in cm
        data['y'] *= 0.1  # now in cm

        # if compact, cosines are scaled by a factor of 30000
        data['cx'] /= 30000
        data['cy'] /= 30000
        # bernloehr clips in his implementation of the reader.
        # we do so here as well. As cx and cy are cosines of angles,
        # values with abs > 1 are not allowed.
        data['cx'] = data['cx'].clip(min=-1., max=1.)
        data['cy'] = data['cy'].clip(min=-1., max=1.)

        data['time'] *= 0.1  # in nanoseconds since first interaction.
        data['zem'] = np.power(10., data['zem'] * 0.001)
        data['photons'] *= 0.01

        return data


class CameraLayout(EventIOObject):
    eventio_type = 1206


class TriggerTime(EventIOObject):
    eventio_type = 1207


class PhotoElectrons(EventIOObject):
    eventio_type = 1208
    from ..var_int import parse_1208

    def __init__(self, header, filehandle):
        super().__init__(header, filehandle)
        self.array_id = header.id // 1000
        self.telescope_id = header.id % 1000

    def parse(self):
        assert_version_in(self, [1, 2, 3])
        self.seek(0)

        pe = {}
        pe['n_pe'] = read_int(self)
        pe['n_pixels'] = read_int(self)
        if self.header.version > 1:
            flags = read_short(self)
        else:
            flags = 0

        pe['non_empty'] = read_int(self)

        data = self.read()

        pe.update(PhotoElectrons.parse_1208(
            data, pe['n_pixels'], pe['non_empty'],
            self.header.version, flags, pe['n_pe']
        ))

        return pe

    def __str__(self):
        return super().__str__() + '(array_id={}, telescope_id={})'.format(
            self.array_id, self.telescope_id,
        )


class EventEnd(EventIOObject):
    eventio_type = 1209

    def parse(self):
        self.seek(0)
        n = read_int(self)
        if n != 273:
            raise WrongSize('Expected 3 floats, but found {}'.format(n))

        return parse_event_end(self.read())

    def __str__(self):
        return super().__str__() + '(event_id={})'.format(self.header.id)


class RunEnd(EventIOObject):
    ''' This Object contains the CORSIKA run end block '''
    eventio_type = 1210

    def parse(self):
        '''
        Read the data in this EventIOObject

        Returns the CORSIKA run end block as arrays of floats.
        No parsing yet, sorry. The meaning is defined in the
        User Guide.
        '''

        self.seek(0)
        n = read_int(self)
        if n != 3:
            raise WrongSize('Expected 3 floats, but found {}'.format(n))
        d = bytearray(273 * 4)
        d[:n * 4] = self.read()
        return parse_run_end(d)


class Longitudinal(EventIOObject):
    ''' This Object contains the CORSIKA longitudinal shower data block '''
    eventio_type = 1211

    def __init__(self, header, filehandle):
        super().__init__(header, filehandle)

    def parse(self):
        '''
        Read the data in this EventIOObject

        Returns the longitudinal shower data block as arrays of floats.
        No parsing yet, sorry. The meaning is defined in the
        User Guide.
        '''
        self.seek(0)
        long = {}
        long['event_id'] = read_int(self)
        long['type'] = read_int(self)
        long['np'] = read_short(self)
        long['nthick'] = read_short(self)
        long['thickstep'] = read_float(self)
        long['data'] = np.frombuffer(
            self.read(4 * long['np'] * long['nthick']),
            dtype='<f4'
        ).reshape(long['np'], long['nthick'])

        return long


class InputCard(EventIOObject):
    ''' This Object contains the CORSIKA steering card '''
    eventio_type = 1212

    def parse(self):
        '''
        Read the data in this EventIOObject

        Returns the  steering card as string.
        '''
        self.seek(0)
        n_strings = read_int(self)
        input_card = bytearray()
        for i in range(n_strings):
            input_card += read_string(self)
            input_card += b'\n'
        return input_card


class AtmosphericProfile(EventIOObject):
    ''' This Object contains the CORSIKA/atmext atmospheric profile '''
    eventio_type = 1216

    def parse(self):
        '''
        Read the data in this EventIOObject
        '''
        self.seek(0)
        assert_max_version(self, 1)
        name = read_var_string(self)
        obslevel = read_double(self)

        table_size = read_unsigned_varint(self)

        # 4 columns, alt_km, rho, rhick, refidx_m1
        table = read_array(self, 'float64', table_size * 4).reshape(table_size, 4)

        n_five_layer = read_unsigned_varint(self)
        if n_five_layer == 5:
            htoa = read_double(self)
            corsika_atmosphere = read_array(self, 'float64', 25).reshape((5, 5))
        else:
            htoa = None
            corsika_atmosphere = None

        return {
            'id': self.header.id,
            'name': name,
            'obslevel': obslevel,
            'altitude_km': table[:, 0],
            'rho': table[:, 1],
            'thickness': table[:, 2],
            'refractive_index_minus_1': table[:, 3],
            'five_layer_atmosphere': corsika_atmosphere,
            'htoa': htoa,
        }
