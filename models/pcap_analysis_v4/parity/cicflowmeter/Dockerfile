# =============================================================
# Official Java CICFlowMeter -- pinned reproducible build.
# Build/run ONLY on x86_64 (linux/amd64). jnetpcap 1.4.r1425
# has no arm64 / macOS native library.
#
#   docker build --platform linux/amd64 -t cfm-pinned .
#   docker run --rm --platform linux/amd64 \
#       -v "$PWD/pcaps":/pcap -v "$PWD/flows":/flow \
#       cfm-pinned /pcap /flow
# =============================================================
FROM --platform=linux/amd64 eclipse-temurin:8-jdk-jammy

# --- native dependency: libpcap (jnetpcap 1.4 needs the 0.8 ABI) ---
# DEBIAN_FRONTEND=noninteractive: without it, some pcap-adjacent packages open a
# debconf prompt and the build hangs forever in CI.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates libpcap0.8 libpcap0.8-dev libpcap-dev tcpdump maven && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# --- pinned CICFlowMeter source ---
# repo   : https://github.com/ahlashkari/CICFlowMeter   (CICFlowMeter-V4.0, "formerly ISCXFlowMeter")
# commit : 98a5ebad0df579cc8b43eedd3421b3ae87699901  (refs/heads/master @ 2026-09-10; repo has NO release tags)
# NOTE   : CSE-CIC-IDS2018 was produced with CICFlowMeter-V3. No pinned V3 revision exists in the
#          official repo. For a V3 comparison use the bundled tree in
#          iPAS/TCPDUMP_and_CICFlowMeter @ 8623d91fa723a865cd8d8d407be7e6493e619e0b
#          (branch comparison_of_cicflowmeter_versions @ 98630d4d2eccba975ed8dbb15e19407c8c27c441).
ARG CFM_COMMIT=98a5ebad0df579cc8b43eedd3421b3ae87699901
RUN git clone https://github.com/ahlashkari/CICFlowMeter.git && \
    cd CICFlowMeter && git checkout ${CFM_COMMIT}

WORKDIR /opt/CICFlowMeter

# --- ensure a libpcap.so.0.8 SONAME exists for jnetpcap 1.4 (jammy ships .so.1) ---
RUN bash -eux -c 'real="$(ldconfig -p | awk -F"=> " "/libpcap\\.so\\.1 /{print \$2; exit}")"; if [ -n "$real" ] && [ ! -e "$(dirname "$real")/libpcap.so.0.8" ]; then ln -s "$(basename "$real")" "$(dirname "$real")/libpcap.so.0.8"; ldconfig; fi; ldconfig -p | grep -i libpcap || true'

# --- install the bundled jnetpcap jar into the local maven repo (build.gradle uses mavenLocal()) ---
RUN bash -eux -c 'mkdir -p /root/.m2; JAR="$(ls jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap*.jar 2>/dev/null | head -1)"; test -n "$JAR"; mvn -q install:install-file -Dfile="$JAR" -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar'

# --- build (vendored Gradle wrapper == gradle 4.2, runs on JDK 8) ---
RUN ./gradlew --no-daemon --console=plain clean build -x test \
 || ./gradlew --no-daemon --console=plain clean assemble -x test \
 || ./gradlew --no-daemon --console=plain clean classes

# --- make the native lib discoverable at runtime ---
ENV LD_LIBRARY_PATH=/opt/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425

# --- entrypoint: cic.cs.unb.ca.ifm.Cmd <pcap_file_or_dir> <out_dir> ---
# defaults inside Cmd.java: flowTimeout=120000000us (120s), activityTimeout=5000000us (5s)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/pcap", "/flow"]
