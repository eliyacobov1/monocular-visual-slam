import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Monocular SLAM Dashboard",
  description: "Visualize homography tracking, pose graphs, and optimization metrics.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
