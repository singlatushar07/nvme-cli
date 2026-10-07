#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Exercise PEL event boundaries through the real CLI without NVMe hardware."""
import json
import os
import re
import struct
import sys
import tempfile
import unittest

from nvme_mock_ipc import MockIPCServer, make_mock_env, resolve_mock_lib_path, run_nvme

NVME_BIN = sys.argv[1] if len(sys.argv) > 1 else 'nvme'
MOCK_LIB = resolve_mock_lib_path()


def pack_event(event_type, payload, vendor=b'', extra_header=b''):
    header = bytearray(24)
    header[0:3] = bytes((event_type, 1, 21 + len(extra_header)))
    struct.pack_into('<HH', header, 20, len(vendor), len(vendor) + len(payload))
    return bytes(header) + extra_header + vendor + payload


def pack_log(events, padding=0):
    header = bytearray(512)
    header[0] = 13
    header[16] = 3
    struct.pack_into('<I', header, 4, len(events))
    struct.pack_into('<Q', header, 8, 512 + sum(map(len, events)) + padding)
    struct.pack_into('<H', header, 18, 492)
    return bytes(header) + b''.join(events) + bytes(padding)


def pack_smart():
    payload = bytearray(512)
    payload[0] = 4
    struct.pack_into('<H', payload, 1, 320)
    payload[3:7] = bytes((100, 50, 133, 4))
    struct.pack_into('<Q', payload, 48, 1091000000)
    struct.pack_into('<Q', payload, 112, 22)
    struct.pack_into('<Q', payload, 128, 7440)
    return bytes(payload)


class PELMockServer(MockIPCServer):
    def __init__(self, sock_path):
        super().__init__(sock_path)
        self.log = b''

    def handle_ioctl(self, conn, fd, request, opcode, nsid,
                     cdw10, cdw11, cdw12, cdw13, cdw14, cdw15, lpo, req_len):
        if opcode == 0x02 and cdw10 & 0xff == 0x0d:
            payload = self.log[lpo:lpo + req_len].ljust(req_len, b'\0')
            self.send_response(conn, 0, payload=payload)
        else:
            self.send_response(conn, 0, payload=bytes(req_len))


class PersistentEventCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nvme-pel-', dir='/tmp')
        self.sock = os.path.join(self.tmp.name, 'ipc.sock')
        self.server = PELMockServer(self.sock)
        self.server.start()
        self.env = make_mock_env(MOCK_LIB, self.sock)

    def tearDown(self):
        self.server.shutdown()
        self.server.join()
        self.tmp.cleanup()

    def check_entries(self, log, expected):
        self.server.log = log
        for fmt in ('normal', 'json'):
            with self.subTest(output_format=fmt):
                result = run_nvme(NVME_BIN, self.env, self.tmp.name, self.tmp.name,
                                  'log', 'persistent-event', '/dev/nvme0',
                                  '-a', '0', '-l', str(len(log)), '-o', fmt)
                self.assertEqual(result.returncode, 0, result.stderr)
                if fmt == 'normal':
                    numbers = re.findall(r'Event Number\s*:?\s+(\d+)', result.stdout)
                    self.assertEqual([int(n) for n in numbers], list(range(len(expected))))
                    for item in expected:
                        if 'nss_hw_err_code' in item:
                            self.assertRegex(result.stdout,
                                             r'Hardware Error Event Code Entry\s*:?\s+'
                                             + str(item['nss_hw_err_code']) + r',')
                else:
                    entries = json.loads(result.stdout)['list_of_event_entries']
                    self.assertEqual(len(entries), len(expected))
                    for index, (actual, wanted) in enumerate(zip(entries, expected)):
                        self.assertEqual(actual['event_number'], index)
                        for key, value in wanted.items():
                            self.assertEqual(actual[key], value, key)

    def test_vendor_information_does_not_shift_following_events(self):
        events = [pack_event(5, b'\x06\0\0\0\x04', bytes(range(8))),
                  pack_event(1, pack_smart(), b'\xaa\xbb\xcc'),
                  pack_event(5, b'\x08\0\0\0')]
        self.check_entries(pack_log(events, padding=128),
                           [{'nss_hw_err_code': 6, 'vu_info_len': 8},
                            {'critical_warning': 4, 'temperature': 320,
                             'percent_used': 133, 'power_on_hours': 7440,
                             'power_cycles': 22, 'data_units_written': 1091000000},
                            {'nss_hw_err_code': 8, 'vu_info_len': 0}])

    def test_last_event_ends_exactly_at_log_boundary(self):
        for vendor in (b'', bytes(range(8))):
            with self.subTest(vendor_length=len(vendor)):
                events = [pack_event(5, b'\x06\0\0\0\x04', vendor),
                          pack_event(5, b'\x08\0\0\0', vendor)]
                self.check_entries(pack_log(events),
                                   [{'nss_hw_err_code': 6}, {'nss_hw_err_code': 8}])

    def test_empty_event_ends_exactly_at_log_boundary(self):
        self.check_entries(pack_log([pack_event(0x7e, b'')]),
                           [{'event_len': 0, 'vu_info_len': 0}])

    def test_extended_header(self):
        events = [pack_event(5, b'\x06\0\0\0\x04', b'VW', b'HHHH'),
                  pack_event(5, b'\x08\0\0\0')]
        self.check_entries(pack_log(events, padding=128),
                           [{'nss_hw_err_code': 6, 'event_header_len': 25,
                             'vu_info_len': 2, 'vs_info_bin': ['VW']},
                            {'nss_hw_err_code': 8}])

    def test_vendor_information_cannot_exceed_event_length(self):
        event = bytearray(pack_event(0x7e, b'\0\0'))
        struct.pack_into('<H', event, 20, 3)
        self.check_entries(pack_log([event], padding=128), [])

    def test_short_event_header_is_not_decoded(self):
        event = bytearray(pack_event(0x7e, b''))
        event[2] = 0
        self.check_entries(pack_log([event], padding=128), [])

    def test_truncated_event_does_not_hide_previous_event(self):
        events = [pack_event(5, b'\x06\0\0\0\x04', b'\x55' * 8),
                  pack_event(1, pack_smart())[:-1]]
        self.check_entries(pack_log(events), [{'nss_hw_err_code': 6}])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
