import { buildApp } from './app.js';

const PORT = parseInt(process.env.PORT ?? '3001');
const HOST = process.env.HOST ?? '0.0.0.0';

const app = await buildApp();
await app.listen({ port: PORT, host: HOST });
console.log(`HomeoRealm API server running at http://${HOST}:${PORT}`);
