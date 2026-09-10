#!/usr/bin/env bash
# CICFlowMeter offline pcap -> flow CSV.
#   entrypoint.sh <pcap_file_or_dir> <out_dir>
set -euo pipefail
IN="${1:-/pcap}"
OUT="${2:-/flow}"
mkdir -p "$OUT"

CP="/opt/CICFlowMeter/build/classes/java/main:/opt/CICFlowMeter/build/resources/main"
# add every jar gradle resolved
for j in $(find /root/.gradle /root/.m2 /opt/CICFlowMeter -name '*.jar' 2>/dev/null); do CP="$CP:$j"; done

echo "CICFlowMeter Cmd  IN=$IN  OUT=$OUT"
java -Xmx4g \
     -Djava.library.path=/opt/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425 \
     -cp "$CP" \
     cic.cs.unb.ca.ifm.Cmd "$IN" "$OUT"

echo "--- produced ---"
ls -la "$OUT"
