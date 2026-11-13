// The runtime must reject overflow/underflow before touching invalid memory.
// This host death test runs under the Linux test/run_tests.sh harness.
#include <csignal>
#include <cstdio>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define CUAD_TAPE_SIZE 2
#include "cuad_rt.h"

static bool valid() {
  double _cuad_tape[CUAD_TAPE_SIZE];
  int _cuad_tp = 0;
  double first = 1.25, second = -0.75, got = 0;
  cuad_push(first);
  cuad_push(second);
  cuad_pop(got);
  if (got != second) return false;
  cuad_pop(got);
  if (got != first || _cuad_tp != 0) return false;
  return true;
}

static void overflow() {
  double _cuad_tape[CUAD_TAPE_SIZE];
  int _cuad_tp = 0;
  cuad_push(1.0);
  cuad_push(2.0);
  cuad_push(3.0);
}

static void underflow() {
  double _cuad_tape[CUAD_TAPE_SIZE];
  int _cuad_tp = 0;
  double got = 0;
  cuad_pop(got);
}

static bool aborts(void (*fn)()) {
  pid_t child = fork();
  if (child < 0) return false;
  if (child == 0) {
    rlimit limit = {0, 0};
    setrlimit(RLIMIT_CORE, &limit);
    fn();
    _exit(0);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) return false;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

int main() {
  bool ok = valid() && aborts(overflow) && aborts(underflow);
  std::printf("tape bounds: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
