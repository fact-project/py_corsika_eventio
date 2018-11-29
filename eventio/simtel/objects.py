''' Implementations of the simtel_array EventIO object types '''
import numpy as np
from ..base import EventIOObject
from ..tools import (
    read_ints,
    read_eventio_string,
    read_from,
    read_utf8_like_signed_int,
    read_array,
    read_time,
    read_vector_of_uint32_scount_differential,
    read_vector_of_uint32_scount_differential_optimized,
)
from ..bits import bool_bit_from_pos


class TelescopeObject(EventIOObject):
    '''
    BaseClass that reads telescope id from header.id and puts it in repr
    '''

    def __init__(self, header, parent):
        super().__init__(header, parent)
        self.telescope_id = header.id

    def __repr__(self):
        return '{}[{}](telescope_id={}, size={}, first_byte={})'.format(
            self.__class__.__name__,
            self.eventio_type,
            self.telescope_id,
            self.header.length,
            self.header.data_field_first_byte
        )


def assert_exact_version(self, supported_version):
    if self.header.version != supported_version:
        raise NotImplementedError(
            (
                'Unsupported version of {name}'
                'only supports version {supported_version}'
                'the given version is {given_version}'
            ).format(
                name=self.__class__.__name__,
                supported_version=supported_version,
                given_version=self.header.version,
            )
        )


def assert_version_in(self, supported_versions):
    if self.header.version not in supported_versions:
        raise NotImplementedError(
            (
                'Unsupported version of {name} '
                'supported versions are: {supported_versions} '
                'the given version is: {given_version} '
            ).format(
                name=self.__class__.__name__,
                supported_versions=supported_versions,
                given_version=self.header.version,
            )
        )


class History(EventIOObject):
    eventio_type = 70


class HistoryCommandLine(EventIOObject):
    eventio_type = 71

    def __init__(self, header, parent):
        super().__init__(header, parent)
        self.timestamp, = read_ints(1, self)

    def parse_data_field(self):
        self.seek(4)  # skip the int, we already read in init
        return read_eventio_string(self)


class HistoryConfig(EventIOObject):
    eventio_type = 72

    def __init__(self, header, parent):
        super().__init__(header, parent)
        self.timestamp, = read_ints(1, self)

    def parse_data_field(self):
        self.seek(4)  # skip the int, we already read in init
        return read_eventio_string(self)


class SimTelRunHeader(EventIOObject):
    eventio_type = 2000
    from .runheader_dtypes import (
        runheader_dtype_part1,
        runheader_dtype_part2
    )

    def __init__(self, header, parent):
        super().__init__(header, parent)
        self.run_id = self.header.id

    def parse_data_field(self):
        '''See write_hess_runheader l. 184 io_hess.c '''
        self.seek(0)
        dt1 = SimTelRunHeader.runheader_dtype_part1

        part1 = read_array(self, dtype=dt1, count=1)[0]
        dt2 = SimTelRunHeader.runheader_dtype_part2(part1['n_telescopes'])
        part2 = read_array(self, dtype=dt2, count=1)[0]

        # rest is two null-terminated strings
        target = read_eventio_string(self)
        observer = read_eventio_string(self)

        result = dict(zip(part1.dtype.names, part1))
        result.update(dict(zip(part2.dtype.names, part2)))
        result['target'] = target
        result['observer'] = observer

        return result


class SimTelMCRunHeader(EventIOObject):
    eventio_type = 2001

    def parse_data_field(self):
        ''' '''
        self.seek(0)
        assert_exact_version(self, 4)

        return {
            'shower_prog_id': read_from('<i', self)[0],
            'shower_prog_vers': read_from('<i', self)[0],
            'shower_prog_start': read_from('<i', self)[0],
            'detector_prog_id': read_from('<i', self)[0],
            'detector_prog_vers': read_from('<i', self)[0],
            'detector_prog_start': read_from('<i', self)[0],
            'obsheight': read_from('<f', self)[0],
            'num_showers': read_from('<i', self)[0],
            'num_use': read_from('<i', self)[0],
            'core_pos_mode': read_from('<i', self)[0],
            'core_range': read_array(self, 'f4', 2),
            'alt_range': read_array(self, 'f4', 2),
            'az_range': read_array(self, 'f4', 2),
            'diffuse': read_from('<i', self)[0],
            'viewcone': read_array(self, 'f4', 2),
            'E_range': read_array(self, 'f4', 2),
            'spectral_index': read_from('<f', self)[0],
            'B_total': read_from('<f', self)[0],
            'B_inclination': read_from('<f', self)[0],
            'B_declination': read_from('<f', self)[0],
            'injection_height': read_from('<f', self)[0],
            'atmosphere': read_from('<i', self)[0],
            'corsika_iact_options': read_from('<i', self)[0],
            'corsika_low_E_model': read_from('<i', self)[0],
            'corsika_high_E_model': read_from('<i', self)[0],
            'corsika_bunchsize': read_from('<f', self)[0],
            'corsika_wlen_min': read_from('<f', self)[0],
            'corsika_wlen_max': read_from('<f', self)[0],
            'corsika_low_E_detail': read_from('<i', self)[0],
            'corsika_high_E_detail': read_from('<i', self)[0],
        }


class SimTelCamSettings(TelescopeObject):
    eventio_type = 2002

    def parse_data_field(self):
        n_pixels, = read_from('<i', self)
        focal_length, = read_from('<f', self)
        pixel_x = read_array(self, count=n_pixels, dtype='float32')
        pixel_y = read_array(self, count=n_pixels, dtype='float32')

        return {
            'telescope_id': self.telescope_id,
            'n_pixels': n_pixels,
            'focal_length': focal_length,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
        }


class SimTelCamOrgan(TelescopeObject):
    eventio_type = 2003

    def parse_data_field(self):
        self.seek(0)
        assert_exact_version(self, supported_version=1)

        num_pixels = read_from('<i', self)[0]
        num_drawers = read_from('<i', self)[0]
        num_gains = read_from('<i', self)[0]
        num_sectors = read_from('<i', self)[0]

        drawer = read_array(self, 'i2', num_pixels)
        card = read_array(
            self, 'i2', num_pixels * num_gains
        ).reshape(num_pixels, num_gains)
        chip = read_array(
            self, 'i2', num_pixels * num_gains
        ).reshape(num_pixels, num_gains)
        channel = read_array(
            self, 'i2', num_pixels * num_gains
        ).reshape(num_pixels, num_gains)

        sectors = []
        for _ in range(num_pixels):
            n = read_from('<h', self)[0]
            sector = read_array(self, 'i2', n)
            # FIXME:
            # according to a comment in the c-sources
            # there is might be an old bug here,
            # which is trailing zeros.
            # is an ascending list of numbes, so any zero
            # after the first position indicates the end of sector.
            #
            # DN: maybe this bug was fixed long ago,
            # so maybe we do not have to account for it here
            # I will check for it in the tests.
            sectors.append(sector)

        sector_type = []
        sector_threshold = []
        sector_pixthresh = []
        for i in range(num_sectors):
            type_, thresh_, pix_thr_ = read_from('<Bff', self)
            sector_type.append(type_)
            sector_threshold.append(thresh_)
            sector_pixthresh.append(pix_thr_)

        return {
            'telescope_id': self.telescope_id,
            'num_drawers': num_drawers,
            'drawer': drawer,
            'card': card,
            'chip': chip,
            'channel': channel,
            'sectors': sectors,
            'sector_type': np.array(sector_type),
            'sector_threshold': np.array(sector_threshold),
            'sector_pixthresh': np.array(sector_pixthresh),
        }




class SimTelPixelset(TelescopeObject):
    eventio_type = 2004
    from .pixelset import dt1, dt2, dt3, dt4

    def parse_data_field(self):
        ''' '''
        self.seek(0)

        p1 = read_array(self, dtype=SimTelPixelset.dt1, count=1)[0]

        dt2 = SimTelPixelset.dt2(num_pixels=p1['num_pixels'])
        p2 = read_array(self, dtype=dt2, count=1)[0]

        dt3 = SimTelPixelset.dt3(num_drawers=p2['num_drawers'])
        p3 = read_array(self, dtype=dt3, count=1)[0]

        nrefshape = read_utf8_like_signed_int(self)
        lrefshape = read_utf8_like_signed_int(self)

        dt4 = SimTelPixelset.dt4(nrefshape, lrefshape)
        p4 = read_array(self, dtype=dt4, count=1)[0]

        return merge_structured_arrays_into_dict([p1, p2, p3, p4])


class SimTelPixelDisable(EventIOObject):
    eventio_type = 2005

    def __init__(self, header, parent):
        super().__init__(header, parent)
        self.telescope_id = header.id

    def parse_data_field(self):
        ''' '''
        self.seek(0)

        assert_exact_version(self, supported_version=0)

        num_trig_disabled, = read_from('<i', self)
        trigger_disabled = read_array(
            self,
            count=num_trig_disabled,
            dtype='i4'
        )
        num_HV_disabled, = read_from('<i', self)
        HV_disabled = read_array(self, count=num_trig_disabled, dtype='i4')

        return {
            'telescope_id': self.telescope_id,
            'num_trig_disabled': num_trig_disabled,
            'trigger_disabled': trigger_disabled,
            'num_HV_disabled': num_HV_disabled,
            'HV_disabled': HV_disabled,
        }


class SimTelCamsoftset(EventIOObject):
    eventio_type = 2006

    def __init__(self, header, parent):
        super().__init__(header, parent)
        self.telescope_id = header.id

    def parse_data_field(self):
        ''' '''
        self.seek(0)
        assert_exact_version(self, supported_version=0)

        dyn_trig_mode, = read_from('<i', self)
        dyn_trig_threshold, = read_from('<i', self)
        dyn_HV_mode, = read_from('<i', self)
        dyn_HV_threshold, = read_from('<i', self)
        data_red_mode, = read_from('<i', self)
        zero_sup_mode, = read_from('<i', self)
        zero_sup_num_thr, = read_from('<i', self)
        zero_sup_thresholds = read_array(self, 'i4', zero_sup_num_thr)
        unbiased_scale, = read_from('<i', self)
        dyn_ped_mode, = read_from('<i', self)
        dyn_ped_events, = read_from('<i', self)
        dyn_ped_period, = read_from('<i', self)
        monitor_cur_period, = read_from('<i', self)
        report_cur_period, = read_from('<i', self)
        monitor_HV_period, = read_from('<i', self)
        report_HV_period, = read_from('<i', self)

        return {
            'telescope_id': self.telescope_id,
            'dyn_trig_mode': dyn_trig_mode,
            'dyn_trig_threshold': dyn_trig_threshold,
            'dyn_HV_mode': dyn_HV_mode,
            'dyn_HV_threshold': dyn_HV_threshold,
            'data_red_mode': data_red_mode,
            'zero_sup_mode': zero_sup_mode,
            'zero_sup_num_thr': zero_sup_num_thr,
            'zero_sup_thresholds': zero_sup_thresholds,
            'unbiased_scale': unbiased_scale,
            'dyn_ped_mode': dyn_ped_mode,
            'dyn_ped_events': dyn_ped_events,
            'dyn_ped_period': dyn_ped_period,
            'monitor_cur_period': monitor_cur_period,
            'report_cur_period': report_cur_period,
            'monitor_HV_period': monitor_HV_period,
            'report_HV_period': report_HV_period,
        }


class SimTelPointingCor(TelescopeObject):
    eventio_type = 2007

    def parse_data_field(self):
        ''' '''
        self.seek(0)
        assert_exact_version(self, supported_version=0)

        function_type, = read_from('<i', self)
        num_param, = read_from('<i', self)
        pointing_param = read_array(self, 'f4', num_param)

        return {
            'telescope_id': self.telescope_id,
            'function_type': function_type,
            'num_param': num_param,
            'pointing_param': pointing_param,
        }


class SimTelTrackSet(TelescopeObject):
    eventio_type = 2008

    def parse_data_field(self):
        tracking_info = {}
        tracking_info['drive_type_az'], = read_from('<h', self)
        tracking_info['drive_type_alt'], = read_from('<h', self)
        tracking_info['zeropoint_az'], = read_from('<f', self)
        tracking_info['zeropoint_alt'], = read_from('<f', self)

        tracking_info['sign_az'], = read_from('<f', self)
        tracking_info['sign_alt'], = read_from('<f', self)
        tracking_info['resolution_az'], = read_from('<f', self)
        tracking_info['resolution_alt'], = read_from('<f', self)
        tracking_info['range_low_az'], = read_from('<f', self)
        tracking_info['range_low_alt'], = read_from('<f', self)
        tracking_info['range_high_az'], = read_from('<f', self)
        tracking_info['range_high_alt'], = read_from('<f', self)
        tracking_info['park_pos_az'], = read_from('<f', self)
        tracking_info['park_pos_alt'], = read_from('<f', self)

        return tracking_info


class SimTelCentEvent(EventIOObject):
    eventio_type = 2009

    def __init__(self, header, parent):
        super().__init__(header, parent)

        if header.version > 2:
            raise IOError('Unsupported CENTEVENT Version: {}'.format(header.version))

        self.global_count = self.header.id

    def parse_data_field(self):

        event_info = {}
        event_info['cpu_time'] = read_time(self)
        event_info['gps_time'] = read_time(self)
        event_info['trigger_pattern'], = read_from('<i', self)
        event_info['data_pattern'], = read_from('<i', self)

        if self.header.version >= 1:
            tels_trigger, = read_from('<h', self)
            event_info['n_triggered_telescopes'] = tels_trigger

            event_info['triggered_telescopes'] = read_array(
                self, count=tels_trigger, dtype='<i2',
            )
            event_info['trigger_times'] = read_array(
                self, count=tels_trigger, dtype='<f4',
            )
            tels_data, = read_from('<h', self)
            event_info['n_telescopes_with_data'] = tels_data
            event_info['telescopes_with_data'] = read_array(
                self, count=tels_data, dtype='<i2'
            )

        if self.header.version >= 2:
            # konrad saves the trigger mask as crazy int, but it uses only 4 bits
            # so it should be indentical to a normal unsigned int with 1 byte
            event_info['teltrg_type_mask'] = read_array(
                self, count=tels_trigger, dtype='uint8'
            )
            assert np.all(event_info['teltrg_type_mask'] < 128), 'Unexpected trigger mask'

            event_info['teltrg_time_by_type'] = {}
            it = zip(event_info['triggered_telescopes'], event_info['teltrg_type_mask'])
            for tel_id, mask in it:
                # trigger times are only written if more than one trigger is there
                if mask not in {0b001, 0b010, 0b100}:
                    event_info['teltrg_time_by_type'][tel_id] = {}
                    for trigger in range(3):
                        if bool_bit_from_pos(mask, trigger):
                            t = read_from('<f', self)[0]
                            event_info['teltrg_time_by_type'][tel_id][trigger] = t

        return event_info


class SimTelTrackEvent(EventIOObject):
    '''Tracking information for a simtel telescope event
    This has no clear type number, since
    Konrad Bernlöhr decided to encode the telescope id into
    the container type as 2100 + tel_id % 100 + 1000 * (tel_id // 100)

    So a container with type 2105 belongs to tel_id 5, 3105 to 105
    '''
    eventio_type = None

    def __init__(self, header, parent):
        self.eventio_type = header.type
        super().__init__(header, parent)
        self.telescope_id = self.type_to_telid(header.type)
        if not self.id_to_telid(header.id) == self.telescope_id:
            raise ValueError('Telescope IDs in type and header do not match')

        self.has_raw = bool(header.id & 0x100)
        self.has_cor = bool(header.id & 0x200)

    def parse_data_field(self):
        dt = []
        if self.has_raw:
            dt.extend([('azimuth_raw', '<f4'), ('altitude_raw', '<f4')])
        if self.has_cor:
            dt.extend([('azimuth_cor', '<f4'), ('altitude_cor', '<f4')])
        return read_array(self, count=1, dtype=dt)[0]

    @staticmethod
    def id_to_telid(eventio_id):
        '''See io_hess.c, l. 2519'''
        return (eventio_id & 0xff) | ((eventio_id & 0x3f000000) >> 16)

    @staticmethod
    def type_to_telid(eventio_type):
        base = eventio_type - 2100
        return 100 * (base // 1000) + base % 1000

    @staticmethod
    def telid_to_type(telescope_id):
        return 2100 + telescope_id % 100 + 1000 * (telescope_id // 100)

    def __repr__(self):
        return '{}[{}](telescope_id={}, size={}, first_byte={})'.format(
            self.__class__.__name__,
            self.eventio_type,
            self.telescope_id,
            self.header.length,
            self.header.data_field_first_byte
        )


class SimTelTelEvent(EventIOObject):
    '''A simtel telescope event
    This has no clear type number, since
    Konrad Bernlöhr decided to encode the telescope id into
    the container type as 2200 + tel_id % 100 + 1000 * (tel_id // 100)

    So a container with type 2205 belongs to tel_id 5, 3205 to 105
    '''
    eventio_type = None

    def __init__(self, header, parent):
        self.eventio_type = header.type
        super().__init__(header, parent)
        self.telescope_id = self.type_to_telid(header.type)
        self.global_count = header.id

    @staticmethod
    def type_to_telid(eventio_type):
        base = eventio_type - 2200
        return 100 * (base // 1000) + base % 1000

    @staticmethod
    def telid_to_type(telescope_id):
        return 2200 + telescope_id % 100 + 1000 * (telescope_id // 100)

    def __repr__(self):
        return '{}[{}](telescope_id={}, size={}, first_byte={})'.format(
            self.__class__.__name__,
            self.eventio_type,
            self.telescope_id,
            self.header.length,
            self.header.data_field_first_byte
        )


class SimTelEvent(EventIOObject):
    eventio_type = 2010


class SimTelTelEvtHead(TelescopeObject):
    eventio_type = 2011

    def parse_data_field(self):
        self.seek(0)
        event_head = {}
        event_head['loc_count'], = read_from('<i', self)
        event_head['glob_count'], = read_from('<i', self)
        event_head['cpu_time'] = read_time(self)
        event_head['gps_time'] = read_time(self)
        t, = read_from('<h', self)
        event_head['trg_source'] = t & 0xff

        if t & 0x100:
            if self.header.version <= 1:
                num_list_trgsect, = read_from('<h', self)
                event_head['list_trgsect'] = read_array(
                    self, dtype='<i2', count=num_list_trgsect
                )
            else:
                num_list_trgsect, = read_utf8_like_signed_int(self)
                event_head['list_trgsect'] = np.array([
                    read_utf8_like_signed_int(self)
                    for _ in range(num_list_trgsect)
                ])
            if self.header.version >= 1 and (t & 0x400):
                event_head['time_trgsect'] = read_array(
                    self, dtype='<f4', count=num_list_trgsect
                )

        if t & 0x200:
            if self.header.version <= 1:
                event_head['num_phys_addr'] = read_from('<h', self)
                event_head['phys_addr'] = read_array(
                    self, dtype='<i2', count=event_head['num_phys_addr']
                )
            else:
                event_head['num_phys_addr'] = read_utf8_like_signed_int(self)
                event_head['phys_addr'] = np.array([
                    read_utf8_like_signed_int(self)
                    for _ in range(event_head['num_phys_addr'])
                ])
        return event_head


class SimTelTelADCSum(EventIOObject):
    eventio_type = 2012


class SimTelTelADCSamp(EventIOObject):
    eventio_type = 2013

    def __init__(self, header, parent):
        super().__init__(header, parent)
        flags_ = header.id
        self._zero_sup_mode = flags_ & 0x1f
        self._data_red_mode = (flags_ >> 5) & 0x1f
        self._list_known = bool((flags_ >> 10) & 0x01)
        if (
            (self._zero_sup_mode != 0 and header.version < 3) or
            self._data_red_mode != 0 or
            self._list_known
        ):
            raise NotImplementedError

        #  !! WTF: raw->zero_sup_mode |= zero_sup_mode << 5;

        self.telescope_id = (flags_ >> 12) & 0xffff

    def parse_data_field(self):
        self.seek(0)
        assert_exact_version(self, supported_version=3)

        args = {
            'num_pixels': read_from('<l', self)[0],
            'num_gains': read_from('<h', self)[0],
            'num_samples': read_from('<h', self)[0],
        }
        if self._zero_sup_mode:
            return self._parse_in_zero_suppressed_mode(**args)
        else:
            return self._parse_in_not_zero_suppressed_mode(**args)

    def _parse_in_zero_suppressed_mode(
        self,
        num_gains,
        num_pixels,
        num_samples,
    ):
        list_size = read_utf8_like_signed_int(self)
        pixel_ranges = []
        for _ in range(list_size):
            start_pixel_id = read_utf8_like_signed_int(self)
            if start_pixel_id < 0:
                pixel_ranges.append(
                    (-start_pixel_id - 1, -start_pixel_id - 1)
                )
            else:
                pixel_ranges.append(
                    (start_pixel_id, read_utf8_like_signed_int(self))
                )

        adc_samples = np.zeros(
            (num_gains, num_pixels, num_samples),
            dtype='u2'
        )
        for i_gain in range(num_gains):
            for pixel_range in pixel_ranges:
                for i_pix in range(*pixel_range):
                    adc_samples[i_gain, i_pix, :] = (
                        read_vector_of_uint32_scount_differential_optimized(
                            self, num_samples
                        )
                    )
        return adc_samples

    def _parse_in_not_zero_suppressed_mode(
        self,
        num_gains,
        num_pixels,
        num_samples,
    ):
        adc_samples = np.zeros(
            (num_gains, num_pixels, num_samples),
            dtype='u2'
        )
        for i_gain in range(num_gains):
            for i_pix in range(num_pixels):
                adc_samples[i_gain, i_pix, :] = (
                    read_vector_of_uint32_scount_differential_optimized(
                        self, num_samples
                    )
                )
        return adc_samples


class SimTelTelImage(EventIOObject):
    eventio_type = 2014

    def parse_data_field(self):
        self.seek(0)
        assert_exact_version(self, supported_version=5)

        flags = self.header.id
        tel_image = {}
        tel_image['flags'] = flags
        tel_image['flags_hex'] = hex(flags)
        tel_image['telescope_id'] = (
            (flags & 0xff) | (flags & 0x3f000000) >> 16
        )
        tel_image['cut_id'] = (flags & 0xff000) >> 12
        tel_image['pixels'] = read_from('<h', self)[0]
        tel_image['num_sat'] = read_from('<h', self)[0]

        # from version 6 on
        # pixels = read_utf8_like_signed_int(self)  # from version 6 on
        # num_sat = read_utf8_like_signed_int(self)

        if tel_image['num_sat'] > 0:
            tel_image['clip_amp'] = read_from('<f', self)[0]

        tel_image['amplitude'] = read_from('<f', self)[0]
        tel_image['x'] = read_from('<f', self)[0]
        tel_image['y'] = read_from('<f', self)[0]
        tel_image['phi'] = read_from('<f', self)[0]
        tel_image['l'] = read_from('<f', self)[0]
        tel_image['w'] = read_from('<f', self)[0]
        tel_image['num_conc'] = read_from('<h', self)[0]
        tel_image['concentration'] = read_from('<f', self)[0]

        if flags & 0x100:
            tel_image['x_err'] = read_from('<f', self)[0]
            tel_image['y_err'] = read_from('<f', self)[0]
            tel_image['phi_err'] = read_from('<f', self)[0]
            tel_image['l_err'] = read_from('<f', self)[0]
            tel_image['w_err'] = read_from('<f', self)[0]

        if flags & 0x200:
            tel_image['skewness'] = read_from('<f', self)[0]
            tel_image['skewness_err'] = read_from('<f', self)[0]
            tel_image['kurtosis'] = read_from('<f', self)[0]
            tel_image['kurtosis_err'] = read_from('<f', self)[0]

        if flags & 0x400:
            # from v6 on this is crazy int
            num_hot = read_from('<h', self)[0]
            tel_image['num_hot'] = num_hot
            tel_image['hot_amp'] = read_array(self, 'f4', num_hot)
            # from v6 on this is array of crazy int
            tel_image['hot_pixel'] = read_array(self, 'i2', num_hot)

        if flags & 0x800:
            tel_image['tm_slope'] = read_from('<f', self)[0]
            tel_image['tm_residual'] = read_from('<f', self)[0]
            tel_image['tm_width1'] = read_from('<f', self)[0]
            tel_image['tm_width2'] = read_from('<f', self)[0]
            tel_image['tm_rise'] = read_from('<f', self)[0]

        return tel_image


class SimTelShower(EventIOObject):
    eventio_type = 2015

    def parse_data_field(self):
        self.seek(0)
        assert_exact_version(self, supported_version=1)

        shower = {}
        result_bits = self.header.id
        shower['result_bits'] = result_bits
        shower['num_trg'] = read_from('<h', self)[0]
        shower['num_read'] = read_from('<h', self)[0]
        shower['num_img'] = read_from('<h', self)[0]
        shower['img_pattern'] = read_from('<i', self)[0]

        if result_bits & 0x01:
            shower['Az'] = read_from('<f', self)[0]
            shower['Alt'] = read_from('<f', self)[0]

        if result_bits & 0x02:
            shower['err_dir1'] = read_from('<f', self)[0]
            shower['err_dir2'] = read_from('<f', self)[0]
            shower['err_dir3'] = read_from('<f', self)[0]

        if result_bits & 0x04:
            shower['xc'] = read_from('<f', self)[0]
            shower['yc'] = read_from('<f', self)[0]

        if result_bits & 0x08:
            shower['err_core1'] = read_from('<f', self)[0]
            shower['err_core2'] = read_from('<f', self)[0]
            shower['err_core3'] = read_from('<f', self)[0]

        if result_bits & 0x10:
            shower['mscl'] = read_from('<f', self)[0]
            shower['mscw'] = read_from('<f', self)[0]

        if result_bits & 0x20:
            shower['err_mscl'] = read_from('<f', self)[0]
            shower['err_mscw'] = read_from('<f', self)[0]

        if result_bits & 0x40:
            shower['energy'] = read_from('<f', self)[0]

        if result_bits & 0x80:
            shower['err_energy'] = read_from('<f', self)[0]

        if result_bits & 0x0100:
            shower['xmax'] = read_from('<f', self)[0]

        if result_bits & 0x0200:
            shower['err_xmax'] = read_from('<f', self)[0]

        return shower


class SimTelPixelTiming(EventIOObject):
    eventio_type = 2016


class SimTelPixelCalib(EventIOObject):
    eventio_type = 2017


class SimTelMCShower(EventIOObject):
    eventio_type = 2020


class SimTelMCEvent(EventIOObject):
    eventio_type = 2021

    def parse_data_field(self):
        ''' '''
        self.seek(0)
        assert_exact_version(self, supported_version=1)

        return {
            'event': self.header.id,
            'shower_num': read_from('<i', self)[0],
            'xcore': read_from('<f', self)[0],
            'ycore': read_from('<f', self)[0],
            # 'aweight': read_from('<f', self),  # only in version 2
        }


class SimTelTelMoni(EventIOObject):
    eventio_type = 2022

    def parse_data_field(self):
        self.seek(0)
        assert_exact_version(self, supported_version=0)

        telescope_id = (
            (self.header.id & 0xff) |
            ((self.header.id & 0x3f000000) >> 16)
        )

        # what: denotes what has changed (since last report?)
        what = ((self.header.id & 0xffff00) >> 8) & 0xffff
        known, = read_from('<h', self)   # C-code used |= instead of = here.
        new_parts, = read_from('<h', self)
        monitor_id, = read_from('<i', self)
        moni_time = read_time(self)

        #  Dimensions of various things
        # version 0
        ns, np, nd, ng = read_from('<hhhh', self)
        # in version 1 this uses crazy 32bit ints
        # ns = read_utf8_like_signed_int(self)
        # np = read_utf8_like_signed_int(self)
        # nd = read_utf8_like_signed_int(self)
        # ng = read_utf8_like_signed_int(self)

        result = {
            'telescope_id': telescope_id,
            'what': what,
            'known': known,
            'new_parts': new_parts,
            'monitor_id': monitor_id,
            'moni_time': moni_time,
        }
        part_parser_args = {
            'num_sectors': ns,
            'num_gains': ng,
            'num_pixels': np,
            'num_drawers': nd,
        }
        result.update(part_parser_args)

        part_parser_map = {
            0x00: self._nothing_changed_here,
            0x01: self._status_only_changed__what_and_0x01,
            0x02: self._counts_and_rates_changed__what_and_0x02,
            0x04: self._pedestal_and_noice_changed__what_and_0x04,
            0x08: self._HV_and_temp_changed__what_and_0x08,
            0x10: self._pixel_scalers_DC_i_changed__what_and_0x10,
            0x20: self._HV_thresholds_changed__what_and_0x20,
            0x40: self._DAQ_config_changed__what_and_0x40,
        }

        for part_id in range(8):
            part_parser = part_parser_map[what & (1 << part_id)]
            result.update(part_parser(**part_parser_args))
        return result

    def _nothing_changed_here(self, **kwargs):
        ''' dummy parser, invoked when this bit is not set '''
        return {}

    def _status_only_changed__what_and_0x01(self, **kwargs):
        return {
            'status_time': read_time(self),
            'status_bits': read_from('<i', self)[0],
        }

    def _counts_and_rates_changed__what_and_0x02(
        self, num_sectors, **kwargs
    ):
        return {
            'trig_time': read_time(self),
            'coinc_count': read_from('<l', self)[0],
            'event_count': read_from('<l', self)[0],
            'trigger_rate': read_from('<f', self)[0],
            'sector_rate': read_array(self, 'f4', num_sectors),
            'event_rate': read_from('<f', self)[0],
            'data_rate': read_from('<f', self)[0],
            'mean_significant': read_from('<f', self)[0],
        }

    def _pedestal_and_noice_changed__what_and_0x04(
        self, num_gains, num_pixels, **kwargs
    ):
        return {
            'ped_noise_time': read_time(self),
            'num_ped_slices': read_from('<h', self)[0],
            'pedestal': read_array(
                self, 'f4', num_gains * num_pixels
            ).reshape((num_gains, num_pixels)),
            'noise': read_array(
                self, 'f4', num_gains * num_pixels
            ).reshape((num_gains, num_pixels)),
        }

    def _HV_and_temp_changed__what_and_0x08(
        self, num_pixels, num_drawers, **kwargs
    ):
        hv_temp_time = read_time(self)
        num_drawer_temp = read_from('<h', self)[0]
        num_camera_temp = read_from('<h', self)[0]
        return {
            'hv_temp_time': hv_temp_time,
            'num_drawer_temp': num_drawer_temp,
            'num_camera_temp': num_camera_temp,
            'hv_v_mon': read_array(self, 'i2', num_pixels),
            'hv_i_mon': read_array(self, 'i2', num_pixels),
            'hv_stat': read_array(self, 'B', num_pixels),
            'drawer_temp': read_array(
                self, 'i2', num_drawers * num_drawer_temp
            ).reshape((num_drawers, num_drawer_temp)),
            'camera_temp': read_array(self, 'i2', num_camera_temp),
        }

    def _pixel_scalers_DC_i_changed__what_and_0x10(
        self, num_pixels, **kwargs
    ):
        return {
            'dc_rate_time': read_time(self),
            'current': read_array(self, 'u2', num_pixels),
            'scaler': read_array(self, 'u2', num_pixels),
        }

    def _HV_thresholds_changed__what_and_0x20(
        self, num_pixels, num_drawers, **kwargs
    ):
        return {
            'hv_thr_time': read_time(self),
            'hv_dac': read_array(self, 'u2', num_pixels),
            'thresh_dac': read_array(self, 'u2', num_drawers),
            'hv_set': read_array(self, 'B', num_pixels),
            'trig_set': read_array(self, 'B', num_pixels),
        }

    def _DAQ_config_changed__what_and_0x40(
        self, **kwargs
    ):
        return {
            'set_daq_time': read_time(self),
            'daq_conf': read_from('<H', self)[0],
            'daq_scaler_win': read_from('<H', self)[0],
            'daq_nd': read_from('<H', self)[0],
            'daq_acc': read_from('<H', self)[0],
            'daq_nl': read_from('<H', self)[0],
        }


class SimTelLasCal(TelescopeObject):
    eventio_type = 2023

    def parse_data_field(self):
        ''' '''
        self.seek(0)
        assert_exact_version(self, supported_version=2)

        num_pixels = read_from('<h', self)[0]
        num_gains = read_from('<h', self)[0]
        lascal_id = read_from('<i', self)[0]
        calib = read_array(
            self, 'f4', num_gains * num_pixels
        ).reshape(num_gains, num_pixels)

        tmp_ = read_array(self, 'f4', num_gains * 2).reshape(num_gains, 2)
        max_int_frac = tmp_[:, 0]
        max_pixtm_frac = tmp_[:, 1]

        tm_calib = read_array(
            self, 'f4', num_gains * num_pixels
        ).reshape(num_gains, num_pixels)

        return {
            'telescope_id': self.telescope_id,
            'lascal_id': lascal_id,
            'calib': calib,
            'max_int_frac': max_int_frac,
            'max_pixtm_frac': max_pixtm_frac,
            'tm_calib': tm_calib,
        }


class SimTelRunStat(EventIOObject):
    eventio_type = 2024


class SimTelMCRunStat(EventIOObject):
    eventio_type = 2025


class SimTelMCPeSum(EventIOObject):
    eventio_type = 2026

    def parse_data_field(self):
        self.seek(0)
        assert_exact_version(self, supported_version=2)

        event = self.header.id
        shower_num = read_from('<i', self)[0]
        num_tel = read_from('<i', self)[0]
        num_pe = read_array(self, 'i4', num_tel)
        num_pixels = read_array(self, 'i4', num_tel)

        # NOTE:
        # I don't see how we can speed this up easily since the length
        # of this thing is not known upfront.

        # pix_pe: a list (running over telescope_id)
        #         of 2-tuples: (pixel_id, pe)
        pix_pe = []
        for n_pe, n_pixels in zip(num_pe, num_pixels):
            if n_pe <= 0 or n_pixels <= 0:
                continue
            non_empty = read_from('<h', self)[0]
            pixel_id = read_array(self, 'i2', non_empty)
            pe = read_array(self, 'i4', non_empty)
            pix_pe.append(pixel_id, pe)

        photons = read_array(self, 'f4', num_tel)
        photons_atm = read_array(self, 'f4', num_tel)
        photons_atm_3_6 = read_array(self, 'f4', num_tel)
        photons_atm_qe = read_array(self, 'f4', num_tel)
        photons_atm_400 = read_array(self, 'f4', num_tel)

        return {
            'event': event,
            'shower_num': shower_num,
            'num_tel': num_tel,
            'num_pe': num_pe,
            'num_pixels': num_pixels,
            'pix_pe': pix_pe,
            'photons': photons,
            'photons_atm': photons_atm,
            'photons_atm_3_6': photons_atm_3_6,
            'photons_atm_qe': photons_atm_qe,
            'photons_atm_400': photons_atm_400,
        }


class SimTelPixelList(EventIOObject):
    eventio_type = 2027

    def parse_data_field(self):
        self.seek(0)
        # even in the prod3b version of Max N the objects
        # of type 2027 seem to be of version 0 only.
        # not sure if version 1 was ever produced.
        assert_exact_version(self, supported_version=0)

        code = self.header.id // 1_000_000
        telescope = self.header.id % 1_000_000

        pixels = read_from('<h', self)
        # in version 1 pixels is a crazy int

        pixel_list = read_array(self, 'i2', pixels)
        # in version 1 pixel_list is an array of crazy int

        return {
            'code': code,
            'telescope': telescope,
            'pixels': pixels,
            'pixel_list': pixel_list,
        }


class SimTelCalibEvent(EventIOObject):
    eventio_type = 2028


def merge_structured_arrays_into_dict(arrays):
    result = dict()
    for array in arrays:
        for name in array.dtype.names:
            result[name] = array[name]
    return result
