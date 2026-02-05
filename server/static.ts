import express, { type Express } from "express";
import fs from "fs";
import path from "path";

export function serveStatic(app: Express) {
  // Look for dist/public relative to the current working directory
  const distPath = path.resolve(process.cwd(), "dist/public");
  
  console.log(`[serveStatic] Current working directory: ${process.cwd()}`);
  console.log(`[serveStatic] Looking for static files at: ${distPath}`);
  console.log(`[serveStatic] Path exists: ${fs.existsSync(distPath)}`);
  
  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`,
    );
  }

  app.use(express.static(distPath));
  console.log(`[serveStatic] Serving static files from ${distPath}`);

  // fall through to index.html if the file doesn't exist
  app.use("*", (_req, res) => {
    const indexPath = path.resolve(distPath, "index.html");
    console.log(`[serveStatic] Serving index.html for path: ${_req.path}`);
    res.sendFile(indexPath);
  });
}
