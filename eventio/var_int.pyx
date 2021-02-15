# cython: language_level=3
import cython
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t
from libc.math cimport NAN

cnp.import_array()


@cython.wraparound(False)  # disable negative indexing
cpdef (uint64_t, uint8_t) unsigned_varint(const uint8_t[:] data, uint64_t offset=0):
    cdef uint8_t length
    cdef uint64_t value
    length = get_length_of_varint(data[offset])
    value = parse_varint(data[offset:offset + length])
    return value, length


@cython.wraparound(False)  # disable negative indexing
cpdef (int64_t, uint8_t) varint(const uint8_t[:] data, uint64_t offset=0):
    cdef uint8_t length
    cdef uint64_t value
    value, length = unsigned_varint(data, offset)
    if (value & 1) == 1:  # Negative number;
        return -(value >> 1) - 1, length
    return value >> 1, length


cpdef uint8_t get_length_of_varint(const uint8_t first_byte):
    if (first_byte & 0x80) == 0:
        return 1
    if (first_byte & 0xc0) == 0x80:
        return 2
    if (first_byte & 0xe0) == 0xc0:
        return 3
    if (first_byte & 0xf0) == 0xe0:
        return 4
    if (first_byte & 0xf8) == 0xf0:
        return 5
    if (first_byte & 0xfc) == 0xf8:
        return 6
    if (first_byte & 0xfe) == 0xfc:
        return 7
    if (first_byte & 0xff) == 0xfe:
        return 8
    return 9


@cython.boundscheck(False)
@cython.wraparound(False)  # disable negative indexing
cpdef uint64_t parse_varint(const uint8_t[:] var_int_bytes):
    cdef uint8_t length = var_int_bytes.shape[0]
    cdef uint64_t v[9]
    cdef uint8_t i  = 0
    for i in range(length):
        v[i] = var_int_bytes[i]

    if length == 1:
        return v[0]

    if length == 2:
        return ((v[0] & 0x3f) <<8) | v[1]

    if length == 3:
        return (
            ((v[0] & 0x1f) << 16)
            | (v[1] << 8)
            | v[2]
        )

    if length == 4:
        return (
            ((v[0] & 0x0f) << 24)
            | (v[1] << 16)
            | (v[2] << 8)
            | v[3]
        )
    if length == 5:
        return (
            ((v[0] & 0x07) << 32)
            | (v[1] << 24)
            | (v[2] << 16)
            | (v[3] << 8)
            | v[4]
        )
    if length == 6:
        return (
            ((v[0] & 0x03) << 40)
            | (v[1] << 32)
            | (v[2] << 24)
            | (v[3] << 16)
            | (v[4] << 8)
            | v[5]
        )
    if length == 7:
        return (
            ((v[0] & 0x01) << 48)
            | (v[1] << 40)
            | (v[2] << 32)
            | (v[3] << 24)
            | (v[4] << 16)
            | (v[5] << 8)
            | v[6]
        )
    if length == 8:
        return (
            (v[1]<<48)
            | (v[2]<<40)
            | (v[3]<<32)
            | (v[4]<<24)
            | (v[5]<<16)
            | (v[6]<<8)
            | v[7]
        )

    return (
        (v[1]<<56)
        | (v[2]<<48)
        | (v[3]<<40)
        | (v[4]<<32)
        | (v[5]<<24)
        | (v[6]<<16)
        | (v[7]<<8)
        | v[8]
    )


@cython.wraparound(False)  # disable negative indexing
cpdef unsigned_varint_array(
    const uint8_t[:] data,
    uint64_t n_elements,
    uint64_t offset = 0,
):
    cdef cnp.npy_intp[1] shape = [n_elements]
    cdef cnp.ndarray[uint64_t, ndim=1] output = cnp.PyArray_SimpleNew(1, shape, cnp.NPY_UINT64)

    cdef uint64_t i
    cdef uint64_t length
    cdef uint64_t pos = offset

    for i in range(n_elements):
        length = get_length_of_varint(data[pos])
        output[i] = parse_varint(data[pos:pos + length])
        pos += length

    return output, pos - offset


@cython.wraparound(False)  # disable negative indexing
cpdef varint_array(
    const uint8_t[:] data,
    uint64_t n_elements,
    uint64_t offset = 0,
):
    cdef cnp.npy_intp[1] shape = [n_elements];
    cdef cnp.ndarray[int64_t, ndim=1] output = cnp.PyArray_SimpleNew(1, shape, cnp.NPY_INT64)

    cdef int64_t val
    cdef uint8_t length

    cdef uint8_t one = 1;
    cdef uint64_t i
    cdef uint64_t pos = offset

    for i in range(n_elements):
        length = get_length_of_varint(data[pos])
        val = parse_varint(data[pos:pos + length])
        pos += length

        if (val & one):
            output[i] = -(val >> one) - one
        else:
            output[i] = val >> one

    return output, pos - offset


@cython.wraparound(False)  # disable negative indexing
cpdef unsigned_varint_arrays_differential(
    const uint8_t[:] data,
    uint64_t n_arrays,
    uint64_t n_elements,
    uint64_t offset = 0,
):
    cdef uint64_t pos = 0
    cdef uint64_t bytes_read = 0
    cdef uint64_t bytes_read_total = 0
    cdef uint64_t i
    cdef uint64_t j

    cdef cnp.npy_intp[2] shape = [n_arrays, n_elements]
    cdef cnp.ndarray[uint32_t, ndim=2] output = cnp.PyArray_SimpleNew(2, shape, cnp.NPY_UINT32)
    cdef uint32_t[:, :] output_view = output
    cdef uint32_t[:] output_view_1d

    for i in range(n_arrays):

        output_view_1d = output_view[i]

        bytes_read = unsigned_varint_array_differential(
            data, output=output_view_1d, offset=offset
        )
        offset += bytes_read
        bytes_read_total += bytes_read

    return output, bytes_read_total


@cython.wraparound(False)  # disable negative indexing
cdef uint64_t unsigned_varint_array_differential(
    const uint8_t[:] data,
    uint32_t[:] output,
    uint64_t offset = 0,
) except -1:

    cdef uint64_t n_elements = output.shape[0]
    cdef int32_t val = 0
    cdef uint64_t i
    cdef uint64_t pos = 0
    cdef uint8_t v0, v1, v2, v3, v4

    for i in range(n_elements):
        v0 = data[pos + offset]
        pos += 1

        if (v0 & 0x80) == 0:  # one byte
            if (v0 & 0x01) == 0:  # positive
                val += v0 >> 1
            else:  # negative
                val -= (v0 >> 1) + 1
        elif (v0 & 0xc0) == 0x80:  # two bytes
            v1 = data[pos + offset]
            pos += 1
            if (v1 & 0x01) == 0:  # positive
                val += ((v0 & 0x3f) << 7) | (v1 >> 1)
            else:  # negative
                val -= (((v0 & 0x3f) << 7) | (v1 >> 1)) + 1
        elif (v0 & 0xe0) == 0xc0:  # three bytes
            v1 = data[pos + offset + 0]
            v2 = data[pos + offset + 1]
            pos += 2

            if (v2 & 0x01) == 0:
                val += (
                    ((v0 & 0x1f) << 15)
                    | (v1 << 7)
                    | (v2 >> 1)
                )
            else:
                val -= (
                    ((v0 & 0x1f) << 15)
                    | (v1 << 7)
                    | (v2 >> 1)
                ) + 1
        elif (v0 & 0xf0) == 0xe0:  # four bytes
            v1 = data[pos + offset + 0]
            v2 = data[pos + offset + 1]
            v3 = data[pos + offset + 2]
            pos += 3
            if (v3 & 0x01) == 0:
                val += (
                    ((v0 & 0x0f) << 23)
                    | (v1 << 15)
                    | (v2 << 7)
                    | (v3 >> 1)
                )
            else:
                val -= (
                    ((v0 & 0x0f) << 23)
                    | (v1 << 15)
                    | (v2 << 7)
                    | (v3 >> 1)
                ) + 1
        elif (v0 & 0xf8) == 0xf0:
            v1 = data[pos + offset + 0]
            v2 = data[pos + offset + 1]
            v3 = data[pos + offset + 2]
            v4 = data[pos + offset + 3]
            pos += 4
            # The format would allow bits 32 and 33 being set but we ignore this here. */
            if (v4 & 0x01) == 0:
                val += (
                    ((v0 & 0x07) << 31)
                    | (v1 << 23)
                    | (v2 << 15)
                    | (v3 << 7)
                    | (v4 >> 1)
                )
            else:
                val -= (
                    ((v0 & 0x07) << 31)
                    | (v1 << 23)
                    | (v2 << 15)
                    | (v3 << 7)
                    | (v4 >> 1)
                ) + 1
        output[i] = val

    return pos



def simtel_pixel_timing_parse_list_type_2(
    const uint8_t[:] data,
    const int16_t[:, :] pixel_list,
    uint32_t n_gains,
    uint32_t n_pixels,
    uint32_t n_types,
    bint with_sum,
    bint glob_only_selected,
    float granularity,
):
    cdef int16_t start, stop
    cdef uint32_t list_index
    cdef uint32_t n_lists = pixel_list.shape[0]
    cdef uint32_t i_pix, i_type
    cdef uint64_t pos = 0
    cdef uint32_t length = 0

    cdef cnp.npy_intp[2] shape = (n_pixels, n_types)
    cdef cnp.ndarray[float, ndim=2] timval = cnp.PyArray_SimpleNew(2, shape, cnp.NPY_FLOAT32)
    cnp.PyArray_FillWithScalar(timval, NAN)

    shape[:] = (n_gains, n_pixels)
    cdef cnp.ndarray[int32_t, ndim=2] pulse_sum_loc = cnp.PyArray_ZEROS(2, shape, cnp.NPY_INT32, False)
    cdef cnp.ndarray[int32_t, ndim=2] pulse_sum_glob = cnp.PyArray_ZEROS(2, shape, cnp.NPY_INT32, False)

    for list_index in range(n_lists):
        start = pixel_list[list_index][0]
        stop = pixel_list[list_index][1]

        for i_pix in range(start, stop + 1):
            for i_type in range(n_types):
                timval[i_pix, i_type] = granularity * (<int16_t*> &data[pos])[0]
                pos += 2

            if with_sum:
                for i_gain in range(n_gains):
                    pulse_sum_loc[i_gain, i_pix], length = varint(data, offset=pos)
                    pos += length

                if glob_only_selected:
                    for i_gain in range(n_gains):
                        pulse_sum_glob[i_gain, i_pix], length = varint(data, offset=pos)
                        pos += length

    if with_sum and pixel_list.shape[0] > 0 and not glob_only_selected:
        for i_gain in range(n_gains):
            for i_pix in range(n_pixels):
                pulse_sum_glob[i_gain, i_pix], length = varint(data, offset=pos)
                pos += length

    return {
        'time': timval,
        'pulse_sum_glob': pulse_sum_glob,
        'pulse_sum_loc': pulse_sum_loc,
    }, pos


def parse_1208(
    const uint8_t[:] data,
    uint32_t n_pixels,
    uint32_t nonempty,
    uint32_t version,
    uint32_t flags,
    uint32_t total_n_pe
):

    cdef uint32_t length

    cdef cnp.npy_intp[1] pixel_shape = [n_pixels]
    cdef cnp.ndarray[int32_t, ndim=1] photoelectrons = cnp.PyArray_ZEROS(1, pixel_shape, cnp.NPY_INT32, False)
    cdef cnp.ndarray[int32_t, ndim=1] photon_counts = None

    cdef cnp.npy_intp[1] pe_shape = [total_n_pe]
    cdef cnp.ndarray[uint32_t, ndim=1] pixel_id = cnp.PyArray_ZEROS(1, pe_shape, cnp.NPY_UINT32, False)
    cdef cnp.ndarray[float, ndim=1] time = cnp.PyArray_ZEROS(1, pe_shape, cnp.NPY_FLOAT32, False)
    cdef cnp.ndarray[float, ndim=1] amplitude = cnp.PyArray_ZEROS(1, pe_shape, cnp.NPY_FLOAT32, False)

    cdef bint has_amplitudes = flags & 1
    cdef bint has_photons = flags & 4

    cdef uint64_t pos = 0
    cdef uint64_t i_pe = 0
    cdef uint32_t i
    cdef int32_t j
    cdef dict result = {}
    cdef float t


    for i in range(nonempty):
        if version > 2:
            pix_id, length = varint(data, offset=pos)
        else:
            pix_id = (<int16_t*> &data[pos])[0]
            length = 2

        pos += length

        n_pe = (<int32_t*> &data[pos])[0]
        pos += 4

        photoelectrons[pix_id] = n_pe

        for j in range(n_pe):
            t = (<float*> &data[pos])[0]
            pixel_id[i_pe + j] = pix_id
            time[i_pe + j] = t
            pos += 4

        if has_amplitudes:
            for j in range(n_pe):
                amplitude[i_pe + j] = (<float*> &data[pos])[0]
                pos += 4

        i_pe += n_pe

    result['photoelectrons'] = photoelectrons
    result['pixel_id'] = pixel_id
    result['time'] = time

    if has_amplitudes:
        result['amplitude'] = amplitude

    if has_photons:
        photons = cnp.PyArray_ZEROS(1, pixel_shape, cnp.NPY_INT32, False)

        nonempty = (<int32_t*> &data[pos])[0]
        pos += 4

        for i in range(nonempty):
            pix_id = (<int16_t*> &data[pos])[0]
            pos += 2

            photons[pix_id] = (<int32_t*> &data[pos])[0]
            pos += 4

        result['photons'] = photons

    return result


cpdef simtel_pixel_timing_parse_list_type_1(
    const uint8_t[:] data,
    const int16_t[:] pixel_list,
    uint32_t n_gains,
    uint32_t n_pixels,
    uint32_t n_types,
    bint with_sum,
    bint glob_only_selected,
    float granularity,
):
    cdef uint32_t start, stop, list_index
    cdef uint32_t pixel_list_length = pixel_list.shape[0]
    cdef uint32_t i, i_gain, i_type, i_pix
    cdef uint32_t length = 0
    cdef int16_t* short_ptr

    cdef cnp.npy_intp[2] shape = (n_pixels, n_types)
    cdef cnp.ndarray[float, ndim=2] timval = cnp.PyArray_SimpleNew(2, shape, cnp.NPY_FLOAT32)
    cnp.PyArray_FillWithScalar(timval, NAN)

    shape[:] = (n_gains, n_pixels)
    cdef cnp.ndarray[int32_t, ndim=2] pulse_sum_loc = cnp.PyArray_ZEROS(2, shape, cnp.NPY_INT32, False)
    cdef cnp.ndarray[int32_t, ndim=2] pulse_sum_glob = cnp.PyArray_ZEROS(2, shape, cnp.NPY_INT32, False)

    cdef uint64_t pos = 0
    for i in range(pixel_list_length):
        i_pix = pixel_list[i]
        for i_type in range(n_types):
            short_ptr = <int16_t*> &(data[pos])
            timval[i_pix, i_type] = granularity * short_ptr[0]
            pos += 2

        if with_sum:
            for i_gain in range(n_gains):
                pulse_sum_loc[i_gain, i_pix], length = varint(data, offset=pos)
                pos += length

            if glob_only_selected:
                for i_gain in range(n_gains):
                    pulse_sum_glob[i_gain, i_pix], length = varint(data, offset=pos)
                    pos += length

    if with_sum and len(pixel_list) > 0 and not glob_only_selected:
        for i_gain in range(n_gains):
            for i_pix in range(n_pixels):
                pulse_sum_glob[i_gain, i_pix], length = varint(
                    data, offset=pos,
                )
                pos += length

    return {
        'time': timval,
        'pulse_sum_glob': pulse_sum_glob,
        'pulse_sum_loc': pulse_sum_loc,
    }, pos



@cython.wraparound(False)  # disable negative indexing
cpdef read_sector_information_v2(
    const uint8_t[:] data,
    uint32_t n_pixels,
    uint64_t offset = 0,
):
    cdef uint32_t i
    cdef uint64_t length
    cdef int64_t n

    cdef list sectors = []
    cdef cnp.ndarray[int64_t, ndim=1] sector

    cdef uint64_t pos = offset
    for i in range(n_pixels):

        n, length = varint(data, offset=pos)
        pos += length

        sector, length = varint_array(data, n, offset=pos)
        pos += length
        sectors.append(sector)

    return sectors, pos - offset
