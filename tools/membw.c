// membw.c — multi-threaded RAM read bandwidth (AVX-512 loads, threads pinned to cores 1..N), and read time of a 4 MB slice split N ways (one layer's CPU experts)
// build/run on the box: gcc -O2 -mavx512f -pthread tools/membw.c -o /tmp/membw && /tmp/membw 12
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char* buf; static size_t total; static int nt; static size_t chunk_off, chunk_len;
static pthread_barrier_t bar; static volatile int phase;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static __m512i sink[64];
static void rd(const char* p, size_t n, int id) {
    __m512i a = _mm512_setzero_si512(), b = a, c = a, d = a;
    for (size_t i = 0; i + 256 <= n; i += 256) {
        a = _mm512_xor_si512(a, _mm512_load_si512(p + i)); b = _mm512_xor_si512(b, _mm512_load_si512(p + i + 64));
        c = _mm512_xor_si512(c, _mm512_load_si512(p + i + 128)); d = _mm512_xor_si512(d, _mm512_load_si512(p + i + 192));
    }
    sink[id] = _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_xor_si512(c, d));
}
static void* worker(void* arg) {
    int id = (int) (size_t) arg;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(1 + id, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    for (;;) {
        pthread_barrier_wait(&bar);
        if (phase < 0) return 0;
        size_t per, off;
        if (phase == 0) { per = total / nt; off = per * id; rd(buf + off, per, id); }
        else { per = (chunk_len / nt) & ~(size_t) 255; off = chunk_off + per * id; rd(buf + off, per, id); }
        pthread_barrier_wait(&bar);
    }
}
int main(int argc, char** argv) {
    nt = argc > 1 ? atoi(argv[1]) : 12; total = (size_t) 2 << 30;
    buf = aligned_alloc(1 << 21, total); memset(buf, 1, total);
    pthread_barrier_init(&bar, 0, nt + 1);
    pthread_t th[64]; for (int i = 0; i < nt; ++i) pthread_create(&th[i], 0, worker, (void*) (size_t) i);
    for (int r = 0; r < 3; ++r) {
        phase = 0; double t0 = now(); pthread_barrier_wait(&bar); pthread_barrier_wait(&bar); double t = now() - t0;
        printf("threads %d: stream 2 GB %.1f GB/s\n", nt, total / t / 1e9);
    }
    chunk_len = 4 << 20; double sum = 0; int reps = 400;
    for (int r = 0; r < reps; ++r) {
        chunk_off = ((size_t) (r * 7919) % 400) * chunk_len;   // a cold 4 MB slice each time
        phase = 1; double t0 = now(); pthread_barrier_wait(&bar); pthread_barrier_wait(&bar); sum += now() - t0;
    }
    printf("threads %d: cold 4 MB slice %.1f us (%.1f GB/s)\n", nt, sum / reps * 1e6, chunk_len / (sum / reps) / 1e9);
    phase = -1; pthread_barrier_wait(&bar);
    long long x = 0; for (int i = 0; i < nt; ++i) x ^= _mm512_reduce_add_epi64(sink[i]);
    printf("(sink %lld)\n", x & 1);
    return 0;
}
