import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Script from "next/script";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <Script
          async
          defer
          data-website-id="6f0216ce-a255-4e5d-bc9d-9c9cdd9c62c7"
          src="https://u.dhr.wtf/script.js"
        ></Script>
        <title>Lecture Chat</title>
        <meta
          name="description"
          content="Stuck during a lecture? Get live transcripts and solve doubts in real-time, without interupting anyone!"
        />
        <meta name="application-name" content="Lecture Chat" />

        <meta property="og:title" content="Lecture Chat" />
        <meta
          property="og:description"
          content="Stuck during a lecture? Get live transcripts and solve doubts in real-time, without interupting anyone!"
        />
        <meta property="og:image" content="/ogimage.png" />
        <meta property="og:image:alt" content="Lecture Chat" />

        <meta name="twitter:title" content="Lecture Chat" />
        <meta
          name="twitter:description"
          content="Lecture Chat is a simple, minimal AI powered note taking app and markdown editor - Built local-first, with cloud sync. It uses AI to help you write and stay productive."
        />
        <meta name="twitter:image" content="/ogimage.png" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:creator" content="@dhravyashah" />

        <link rel="icon" type="image/png" href="/logo.png" />
        <meta name="theme-color" content="#54CFDF" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Lecture Chat"></meta>
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
