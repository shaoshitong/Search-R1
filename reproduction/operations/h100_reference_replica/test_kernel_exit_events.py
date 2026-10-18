import signal
import struct
import unittest
from .kernel_exit_events import NL,CN,COMMON,EXIT,decode_packet


def message(what,payload):
    event=COMMON.pack(what,2,123456)+payload
    connector=CN.pack(1,1,3,0,len(event),0)+event
    return NL.pack(NL.size+len(connector),3,0,0,0)+connector


class ExitPacketTests(unittest.TestCase):
    def test_raw_wait_status_not_sigchld_is_decoded(self):
        for raw,code in [(0,0),(7<<8,7)]:
            event=decode_packet(message(EXIT,struct.pack('=IIIIII',100,100,raw,signal.SIGCHLD,1,1)))[0]
            self.assertEqual(event['exit_code'],code)
            self.assertEqual(event['parent_notification_signal'],signal.SIGCHLD)
            self.assertIsNone(event['termination_signal'])

    def test_signal_and_nonleader_identity_are_retained(self):
        event=decode_packet(message(EXIT,struct.pack('=IIII',101,100,signal.SIGTERM,signal.SIGCHLD)))[0]
        self.assertEqual(event['termination_signal'],signal.SIGTERM)
        self.assertIsNone(event['exit_code']);self.assertNotEqual(event['pid'],event['tgid'])

    def test_ack_and_truncation(self):
        self.assertEqual(decode_packet(message(0,struct.pack('=I',0)))[0],dict(kind='ack',error=0))
        with self.assertRaises(ValueError):decode_packet(message(EXIT,b'bad'))
        with self.assertRaises(ValueError):decode_packet(b'bad')


if __name__=='__main__':unittest.main()
