import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 5,
  duration: '30s',
  thresholds: {
    http_req_failed: ['rate<0.05'],
    http_req_duration: ['p(95)<500'],
  },
};

const BASE_URL = __ENV.API_URL || 'http://127.0.0.1:8000';
const API_KEY = __ENV.API_KEY || '';
const headers = API_KEY ? { 'x-api-key': API_KEY } : {};

export default function () {
  // Ready
  let r1 = http.get(`${BASE_URL}/ready`);
  check(r1, { 'ready 200': (r) => r.status === 200 });

  // Metrics
  let r2 = http.get(`${BASE_URL}/metrics`);
  check(r2, { 'metrics 200': (r) => r.status === 200 });

  // Ask
  let r3 = http.post(`${BASE_URL}/ask`, JSON.stringify({ query: 'test question' }), {
    headers: { 'Content-Type': 'application/json', ...headers },
  });
  check(r3, { 'ask 200': (r) => r.status === 200 });

  sleep(0.5);
}
