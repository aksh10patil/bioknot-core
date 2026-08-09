"use client";

import dynamic from 'next/dynamic';

const BioKnotDashboard = dynamic(
  () => import('@/components/BioKnotDashboard'),
  { ssr: false }
);

export default function Home() {
  return <BioKnotDashboard />;
}
