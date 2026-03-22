import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Shorts Creator Studio",
  description:
    "AI video processing with a live clip generation engine, pipeline feedback, and creator controls.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
