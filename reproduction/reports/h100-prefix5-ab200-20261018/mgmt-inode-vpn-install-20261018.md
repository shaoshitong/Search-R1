# mgmt-host iNode installation and TLS diagnosis

Human authorized H3C iNode from the supplied gateway and explicitly requested stopping mgmt-host Tailscale and UniVPN on 2026-10-18. This changes observation transport only; native training was not stopped or resubmitted. Do not automatically restart those VPN clients under an old route replay.

- Host: mgmt-host, Ubuntu 24.04 ARM64.
- Gateway: https://183.253.62.193:2000. Its client manifest points to the H3C official Linux ARM package, download id 7684850.
- Original package: 17,841,945 bytes, SHA256 `bd021799f46b13a4e107ea44859980bc9c687863f97352c030348f0dd23e92e4`.
- Vendor archive retained remotely under `/home/lichen/vpn-client-install-20261018`; no binaries on Mac or Git.
- Installed at `/opt/H3C/iNodeClient`. GTK2 installed from Ubuntu apt; libpng12 ARM library extracted from Ubuntu Ports into iNode's private libs directory. Ubuntu vendor init script supplied explicitly because the installer selects it only for Kylin/UOS. `inode.service` is active. Original installer retained in the downloaded archive.
- Task-only GUI display :197, Xauthority private under `/run/h3c-inode-task`; no TCP X11 listener. Credential saving checkbox was disabled. Generic UI helper accepts runtime actions via stdin and never writes them to source or report.
- UniVPN ARM64 had previously been installed from its vendor and failed before authentication; its interactive process exited. No UniVPN process remained when the human requested stopping it.
- Actual at 08:39:14 UTC: `tailscaled.service` stopped/inactive, no UniVPN processes, only H3C authentication/UI components active.
- iNode attempted the supplied gateway at 08:34:58 UTC and again after stopping other VPN clients at 08:39:46 UTC. Both failed during gateway-parameter negotiation with OpenSSL `0x1409442e`: `tlsv1 alert protocol version`, before account authentication.
- Independent protocol probes at 08:38:05 UTC: TLS1.0 rejected by gateway with alert 70; TLS1.1 and TLS1.2 successfully negotiated. These probes did not authenticate or establish a VPN tunnel.
- This old ARM archive was customized in August 2022. Its observed connection properties offer no TLS version selector. Need a compatible iNode Linux ARM build with TLS1.2 (or documented vendor configuration); do not silently alter vendor binaries or lower server policy.
- H3C official same-symptom diagnosis: https://knowledge.h3c.com/Theme/details/195300 . Official recommendation is matching client protocol version; newer clients expose TLS1.2 in connection properties.

**Status:** installation and requested competing-client stops completed; H3C VPN tunnel not established. Existing H100 training progress remains unknown beyond the last successful observation. No password, private key, full VPN session data, model or environment is included here.
