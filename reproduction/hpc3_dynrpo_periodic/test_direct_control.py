"""Local fake interactive SSH checks real terminal descriptors and bounded I/O."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from reproduction.hpc3_dynrpo_periodic import direct_control as direct


class DirectControlTests(unittest.TestCase):
    def test_strict_explicit_route_and_no_password_in_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            hosts=Path(temporary)/'known_hosts';hosts.write_text('existing verified host key')
            with mock.patch.object(direct,'KNOWN_HOSTS',hosts),mock.patch.object(
                    direct.subprocess,'check_output',return_value='10.4.166.96\n'):
                args=direct.command_args('hostname','mou176','10.4.166.96')
                for flag in ('-oProxyCommand=none','-oProxyJump=none','-oStrictHostKeyChecking=yes',
                             '-oHostKeyAlias='+direct.HOST_ALIAS,'-oGlobalKnownHostsFile=/dev/null'):
                    self.assertIn(flag,args)
                self.assertEqual(args[-2],'mou176@10.120.81.30')
                with self.assertRaises(ValueError):direct.command_args('hostname','mou176','10.4.166.97')
                with self.assertRaises(ValueError):direct.command_args('hostname','other','10.4.166.96')

    def test_real_standard_terminals_password_prompt_and_output_redaction(self):
        program=r'''
import os,sys,termios
assert all(os.isatty(fd) for fd in (0,1,2))
with open('/dev/tty','r+b',buffering=0) as tty:
 state=termios.tcgetattr(tty.fileno());state[3]&=~termios.ECHO
 termios.tcsetattr(tty.fileno(),termios.TCSANOW,state)
 tty.write(b"fake@host's password:")
 password=bytearray()
 while True:
  c=tty.read(1)
  if c==b'\n':break
  password.extend(c)
 sys.stdout.buffer.write(b'accepted:'+password+b'\n');sys.stdout.flush()
 sys.stderr.buffer.write(b'merged-stderr\n');sys.stderr.flush()
'''
        secret=b'LOCAL_ONLY_TEST_SECRET'
        result=direct._execute([sys.executable,'-c',program],b'',secret,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn(b'accepted:[REDACTED]\n',result.stdout)
        self.assertIn(b'merged-stderr\n',result.stdout)
        self.assertNotIn(secret,result.stdout+result.stderr)
        self.assertNotIn(secret,b' '.join(x.encode() for x in result.args))
        self.assertNotIn(secret,b' '.join(x.encode() for x in os.environ.values()))
        self.assertNotIn(b'password:',result.stdout)

    def test_nonempty_payload_timeout_and_output_limit_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'command-only'):
            direct._execute([sys.executable,'-c','pass'],b'unsupported',b'test')
        result=direct._execute([sys.executable,'-c','import time;time.sleep(2)'],b'',b'test',
                               timeout=2,prompt_timeout=.1)
        self.assertEqual(result.returncode,124);self.assertIn(b'prompt timed out',result.stderr)
        result=direct._execute([sys.executable,'-c','print("x"*1000)'],b'',b'test',
                               timeout=2,output_limit=100)
        self.assertEqual(result.returncode,124);self.assertIn(b'small-metadata limit',result.stderr)


if __name__=='__main__':unittest.main()
