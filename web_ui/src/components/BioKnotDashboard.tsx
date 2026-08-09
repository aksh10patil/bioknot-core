"use client";

import React, { useEffect, useRef, useState } from 'react';

interface TopologyReport {
  num_residues: number;
  max_diameter: number;
  estimated_crossing_number: number;
  is_entangled: boolean;
  knot_type?: string;
}

interface DomainResult {
  domainIndex: number;
  startResidue: number;
  endResidue: number;
  sequence: string;
  report: TopologyReport;
  pdbData: string;
}

export default function BioKnotDashboard() {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [wasmReady, setWasmReady] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [activeReport, setActiveReport] = useState<TopologyReport | null>(null);
  const [execTime, setExecTime] = useState<string>("");
  const [wasmModule, setWasmModule] = useState<any>(null);
  const [domainResults, setDomainResults] = useState<DomainResult[]>([]);
  
  // Default: Human Ubiquitin (76 aa)
  const [fastaSeq, setFastaSeq] = useState<string>(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
  );
  const [pdbString, setPdbString] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [statusInfo, setStatusInfo] = useState<string>("");

  useEffect(() => {
    async function loadWasm() {
      try {
        const wasm = await import('../../public/wasm/bioknot_wasm.js');
        await wasm.default();
        setWasmModule(wasm);
        setWasmReady(true);
      } catch (err) {
        console.error("WASM Load Error:", err);
        setErrorMessage("Failed to load Rust WASM binary.");
      }
    }
    loadWasm();
  }, []);

  const parseCaCoordinates = (pdbText: string): number[] => {
    const lines = pdbText.split('\n');
    const coords: number[] = [];

    for (const line of lines) {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        const atomName = line.substring(12, 16).trim();
        if (atomName === 'CA') {
          const x = parseFloat(line.substring(30, 38).trim());
          const y = parseFloat(line.substring(38, 46).trim());
          const z = parseFloat(line.substring(46, 54).trim());

          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            coords.push(x, y, z);
          }
        }
      }
    }
    return coords;
  };

  const render3DStructure = (pdbContent: string) => {
    if (viewerRef.current && typeof window !== 'undefined') {
      import('3dmol').then((ThreeDMol) => {
        viewerRef.current!.innerHTML = '';
        const viewer = ThreeDMol.createViewer(viewerRef.current, {
          backgroundColor: '#0d1117',
        });

        viewer.addModel(pdbContent, "pdb");
        viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
        viewer.zoomTo();
        viewer.render();
      });
    }
  };

  const foldSequenceChunk = async (seq: string): Promise<string> => {
    const response = await fetch('https://api.esmatlas.com/foldSequence/v1/pdb/', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: seq,
    });

    if (!response.ok) {
      throw new Error(`ESMFold API Error (Status ${response.status})`);
    }

    return await response.text();
  };

  const handlePredict3DAndAnalyze = async () => {
    const cleanSeq = fastaSeq.replace(/[^A-Z]/gi, '').toUpperCase();
    if (!cleanSeq) {
      setErrorMessage("Please enter a valid FASTA amino acid sequence.");
      return;
    }

    setErrorMessage("");
    setStatusInfo("");
    setIsPredicting(true);
    setActiveReport(null);
    setDomainResults([]);

    const DOMAIN_LIMIT = 400; // ESMFold API limit threshold

    try {
      if (cleanSeq.length <= DOMAIN_LIMIT) {
        // Single Domain Pipeline
        setStatusInfo("Folding full sequence via ESMFold API...");
        const pdbData = await foldSequenceChunk(cleanSeq);
        setPdbString(pdbData);
        render3DStructure(pdbData);

        const caCoords = parseCaCoordinates(pdbData);
        if (wasmReady && wasmModule && caCoords.length > 0) {
          const t0 = performance.now();
          const coordsFloat64 = new Float64Array(caCoords);
          const res = wasmModule.analyze_topology_wasm(coordsFloat64) as TopologyReport;
          const t1 = performance.now();

          if (typeof (res as any).knot_type === 'function') {
            res.knot_type = (res as any).knot_type();
          }

          setExecTime((t1 - t0).toFixed(4));
          setActiveReport(res);
        }
      } else {
        // Multi-Domain Pipeline (> 400 AA)
        setStatusInfo(`Sequence length (${cleanSeq.length} AA) exceeds 400 residues. Initiating automated domain segmentation...`);
        
        const windowSize = 300;
        const step = 200;
        const results: DomainResult[] = [];

        for (let i = 0; i < cleanSeq.length; i += step) {
          const start = i;
          const end = Math.min(i + windowSize, cleanSeq.length);
          const domainSeq = cleanSeq.substring(start, end);

          if (domainSeq.length < 50) break; // Skip trailing fragments < 50 AA

          setStatusInfo(`Folding Domain Segment ${results.length + 1}: Residues ${start + 1}–${end}...`);
          
          const pdbData = await foldSequenceChunk(domainSeq);
          const caCoords = parseCaCoordinates(pdbData);

          if (wasmReady && wasmModule && caCoords.length > 0) {
            const coordsFloat64 = new Float64Array(caCoords);
            const res = wasmModule.analyze_topology_wasm(coordsFloat64) as TopologyReport;

            if (typeof (res as any).knot_type === 'function') {
              res.knot_type = (res as any).knot_type();
            }

            results.push({
              domainIndex: results.length + 1,
              startResidue: start + 1,
              endResidue: end,
              sequence: domainSeq,
              report: res,
              pdbData,
            });
          }

          if (end === cleanSeq.length) break;
        }

        setDomainResults(results);
        if (results.length > 0) {
          // Render first domain by default
          setPdbString(results[0].pdbData);
          render3DStructure(results[0].pdbData);
          setActiveReport(results[0].report);
        }
        setStatusInfo(`Completed topological domain segmentation across ${results.length} domains.`);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || "Failed to execute sequence folding.");
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <header className="mb-8 border-b border-slate-800 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-emerald-400">
            BioKnot-Core Structural Topology Studio
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Deep Learning Structure Inference (ESMFold) + Rust WASM Topological Analysis
          </p>
        </div>
        <span className={`px-4 py-2 rounded-full text-xs font-semibold ${wasmReady ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-amber-950 text-amber-400 border border-amber-800'}`}>
          {wasmReady ? 'Rust WASM Engine Active' : 'Initializing Module...'}
        </span>
      </header>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-8">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Target Sequence Input</h2>
        <div className="flex flex-col gap-4">
          <div>
            <textarea
              rows={4}
              value={fastaSeq}
              onChange={(e) => setFastaSeq(e.target.value)}
              placeholder="Paste FASTA amino acid sequence..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm font-mono text-emerald-400 focus:outline-none focus:border-emerald-500"
            />
            <div className="text-xs text-slate-500 mt-1 flex justify-between">
              <span>Sequence Length: {fastaSeq.replace(/[^A-Z]/gi, '').length} AA</span>
              <span>Long sequence (&gt;400 AA) auto-segmentation enabled</span>
            </div>
          </div>

          {statusInfo && (
            <div className="text-emerald-400 text-xs bg-emerald-950/40 border border-emerald-800/50 p-3 rounded-lg font-mono">
              ℹ️ {statusInfo}
            </div>
          )}

          {errorMessage && (
            <div className="text-rose-400 text-xs bg-rose-950/40 border border-rose-800/50 p-3 rounded-lg">
              ❌ {errorMessage}
            </div>
          )}

          <button
            onClick={handlePredict3DAndAnalyze}
            disabled={!wasmReady || isPredicting}
            className="w-full py-3 px-6 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 text-white font-medium rounded-lg transition-colors shadow-lg flex justify-center items-center gap-2"
          >
            {isPredicting ? (
              <>
                <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
                Processing ESMFold & Evaluating Invariants...
              </>
            ) : (
              'Predict 3D Structure & Classify Topological Knots'
            )}
          </button>
        </div>
      </div>

      {domainResults.length > 0 && (
        <div className="mb-8 bg-slate-900 border border-slate-800 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 uppercase mb-3">Segmented Domain Analysis</h3>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {domainResults.map((dom) => (
              <button
                key={dom.domainIndex}
                onClick={() => {
                  setPdbString(dom.pdbData);
                  render3DStructure(dom.pdbData);
                  setActiveReport(dom.report);
                }}
                className={`px-4 py-2 rounded-lg text-xs font-semibold border transition-all ${
                  activeReport === dom.report
                    ? 'bg-emerald-600 border-emerald-400 text-white'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                Domain {dom.domainIndex} ({dom.startResidue}–{dom.endResidue} AA): {dom.report.knot_type}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h2 className="text-lg font-semibold mb-3 text-slate-200">Predicted 3D Backbone (ESMFold)</h2>
          <div ref={viewerRef} className="w-full h-[450px] rounded-lg overflow-hidden border border-slate-800 relative bg-slate-950 flex items-center justify-center text-slate-600 text-xs">
            {!pdbString && "3D Canvas Idle. Enter sequence and run prediction."}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-200 mb-6">WASM Knot Analytics Engine</h2>

            {activeReport ? (
              <div className="space-y-4">
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 flex justify-between items-center">
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-wider">Identified Knot Type</div>
                    <div className="text-2xl font-extrabold text-emerald-400 mt-1">
                      {activeReport.knot_type || "Unknot (0_1)"}
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${activeReport.is_entangled ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-slate-800 text-slate-300'}`}>
                    {activeReport.is_entangled ? 'KNOTTED' : 'UNKNOTTED'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="text-slate-500 text-xs">Backbone C-alpha Atoms</div>
                    <div className="text-xl font-bold mt-1">{activeReport.num_residues}</div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="text-slate-500 text-xs">Projected Crossings</div>
                    <div className="text-xl font-bold mt-1">{activeReport.estimated_crossing_number}</div>
                  </div>
                </div>

                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex justify-between items-center">
                  <div>
                    <div className="text-slate-500 text-xs">Spatial Diameter (D_max)</div>
                    <div className="text-base font-semibold mt-1">{activeReport.max_diameter.toFixed(2)} Å</div>
                  </div>
                  {execTime && (
                    <div className="text-right">
                      <div className="text-slate-500 text-xs">WASM Compute Time</div>
                      <div className="text-xs font-mono text-emerald-400 mt-1">{execTime} ms</div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-slate-500 text-sm italic">
                Awaiting sequence prediction execution.
              </div>
            )}
          </div>

          <footer className="text-xs text-slate-600 mt-6 border-t border-slate-800/50 pt-4">
            BioKnot-Core Systems Architecture — ESMFold Deep Learning + Rust WebAssembly
          </footer>
        </div>
      </div>
    </div>
  );
}
