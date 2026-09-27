import type { NextConfig } from "next";

// The browser calls same-origin /api/backend/* (see lib/api.ts) and Next.js proxies
// those requests to the real backend here. This keeps the backend URL out of the
// client bundle and avoids browser CORS entirely.
//
// This app is deployed on Vercel; the backend runs on Modal (see Docs/DEPLOYMENT.md).
// Set BACKEND_URL in the Vercel project's Environment Variables to the backend's
// PUBLIC https URL, e.g. https://<workspace>--namtheg-backend-fastapi-app.modal.run.
// A bare hostname is assumed public and gets https://. Defaults to local dev.
//
// Rewrites with an absolute external destination are served by Vercel's routing
// layer rather than a Serverless Function, so the Hobby function timeout does not
// apply to these proxied calls — which matters for runs that take a while.
//
// Read at build time, so redeploy after changing it.
const rawBackend = process.env.BACKEND_URL ?? "http://localhost:8000";
const backendOrigin = (
  /^https?:\/\//i.test(rawBackend) ? rawBackend : `https://${rawBackend}`
).replace(/\/+$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backendOrigin}/:path*`,
      },
    ];
  },
};

export default nextConfig;
