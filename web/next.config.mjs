const apiUrl = process.env.REPOWREN_API_URL ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Cold GraphRAG planning can exceed Next's 30-second development proxy
    // default on a 4 GB GPU. Match the vLLM client's request timeout.
    proxyTimeout: 180_000,
  },
  async rewrites() {
    return [
      {
        source: "/api/repowren/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
