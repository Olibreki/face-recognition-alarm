#include <stdio.h>
#include <stdlib.h>

// This launcher activates the venv and then runs the Python detection pipeline.
// LV_HOME can override the project root (defaults to the current working directory).
int main() {
    const char* lv_home = getenv("LV_HOME");
    if (!lv_home || lv_home[0] == '\0') {
        lv_home = ".";
    }

    char command[1024];
    snprintf(command, sizeof(command),
             "bash -c 'source %s/LVPython/bin/activate && python %s/live_feed_detect_face.py'",
             lv_home, lv_home);

    int ret = system(command);
    return ret;
}
