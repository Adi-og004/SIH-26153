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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates libpcap0.8 libpcap-dev tcpdump wireshark-common && \
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

# --- install the bundled jnetpcap jar into the local gradle/maven cache ---
# (build.gradle: org.jnetpcap:jnetpcap:1.4.1 ; native = jnetpcap/linux/jnetpcap-1.4.r1425/*.so)
RUN mkdir -p /root/.m2 && \
    (command -v mvn >/dev/null 2>&1 || (apt-get update && apt-get install -y maven && rm -rf /var/lib/apt/lists/*)) && \
    mvn -q install:install-file \
        -Dfile=jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar \
        -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar

# --- build (Gradle wrapper is vendored) ---
RUN ./gradlew --no-daemon clean build -x test || ./gradlew --no-daemon clean assemble -x test

# --- make the native lib discoverable at runtime ---
ENV LD_LIBRARY_PATH=/opt/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425

# --- entrypoint: cic.cs.unb.ca.ifm.Cmd <pcap_file_or_dir> <out_dir> ---
# defaults inside Cmd.java: flowTimeout=120000000us (120s), activityTimeout=5000000us (5s)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/pcap", "/flow"]
