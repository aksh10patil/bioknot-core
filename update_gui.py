html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BioKnot-Core | Native AI & WASM Structure Suite</title>
    <!-- Molstar Viewer CSS & JS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@3.38.0/build/viewer/molstar.css" />
    <script src="https://cdn.jsdelivr.net/npm/molstar@3.38.0/build/viewer/molstar.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .container { max-width: 1100px; margin: 0 auto; background: #1e293b; padding: 25px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; font-size: 1.8rem; margin-bottom: 5px; }
        p { color: #94a3b8; font-size: 0.95rem; }
        textarea { width: 100%; height: 110px; background: #0f172a; border: 1px solid #334155; color: #f8fafc; padding: 12px; border-radius: 8px; font-family: monospace; box-sizing: border-box; resize: vertical; }
        .btn-group { display: flex; gap: 10px; margin-top: 15px; }
        button { background: #0284c7; color: white; border: none; padding: 12px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #0369a1; }
        button.secondary { background: #475569; }
        button.secondary:hover { background: #334155; }
        #results { margin-top: 20px; padding: 15px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; display: none; }
        .knot-badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; margin-top: 5px; }
        .unknot { background: #15803d; color: white; }
        .knot { background: #b91c1c; color: white; }
        #viewport { width: 100%; height: 500px; position: relative; margin-top: 15px; border-radius: 8px; overflow: hidden; background: #000; display: none; }
        pre { white-space: pre-wrap; word-wrap: break-word; font-size: 0.8rem; color: #94a3b8; max-height: 200px; overflow-y: auto; background: #020617; padding: 10px; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BioKnot-Core Platform</h1>
        <p>Hybrid Topological Protein Knot Detection & WASM 3D Structure Bridge</p>
        
        <label for="seqInput"><strong>Amino Acid Sequence (FASTA or Raw):</strong></label>
        <textarea id="seqInput" placeholder="Paste amino acid sequence here..."></textarea>
        
        <div class="btn-group">
            <button onclick="analyzeTopology()">Analyse Topology (Native AI)</button>
            <button class="secondary" onclick="predict3D()">Generate 3D C-Alpha Nodes</button>
        </div>

        <div id="results">
            <h3>Inference Output</h3>
            <div id="knotOutput"></div>
            <div id="viewport"></div>
            <details style="margin-top: 10px;">
                <summary style="cursor:pointer; color:#38bdf8;">View Raw Coordinates / JSON Payload</summary>
                <pre id="jsonOutput"></pre>
            </details>
        </div>
    </div>

    <script>
        let viewer = null;

        async function analyzeTopology() {
            const seq = document.getElementById('seqInput').value.replace(/[\r\n\s]/g, '');
            if (!seq) { alert('Please enter a valid amino acid sequence.'); return; }

            const resDiv = document.getElementById('results');
            const knotOut = document.getElementById('knotOutput');
            const jsonOut = document.getElementById('jsonOutput');
            document.getElementById('viewport').style.display = 'none';

            resDiv.style.display = 'block';
            knotOut.innerHTML = '<em>Running Native Transformer Inference...</em>';
            jsonOut.textContent = '';

            try {
                const response = await fetch('/predict_custom', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence: seq })
                });
                const data = await response.json();

                const isUnknot = data.predicted_knot && data.predicted_knot.includes('Unknot');
                const badgeClass = isUnknot ? 'unknot' : 'knot';

                knotOut.innerHTML = `
                    <div><strong>Evaluated Length:</strong> ${data.evaluated_length} AA (Total: ${data.sequence_length} AA)</div>
                    <div><strong>Predicted Topology:</strong> <span class="knot-badge ${badgeClass}">${data.predicted_knot}</span></div>
                    <div><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</div>
                `;
                jsonOut.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                knotOut.innerHTML = '<span style="color: #ef4444;">Inference Error: Connection failed or server offline.</span>';
            }
        }

        async function predict3D() {
            const seq = document.getElementById('seqInput').value.replace(/[\r\n\s]/g, '');
            if (!seq) { alert('Please enter a valid amino acid sequence.'); return; }

            const resDiv = document.getElementById('results');
            const knotOut = document.getElementById('knotOutput');
            const jsonOut = document.getElementById('jsonOutput');
            const viewport = document.getElementById('viewport');

            resDiv.style.display = 'block';
            viewport.style.display = 'block';
            knotOut.innerHTML = '<em>Executing Rigid Kabsch Tiling Bridge...</em>';
            jsonOut.textContent = '';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence: seq })
                });
                const data = await response.json();

                if (data.error) {
                    knotOut.innerHTML = `<span style="color: #ef4444;">Error: ${data.error}</span>`;
                    viewport.style.display = 'none';
                    return;
                }

                knotOut.innerHTML = `
                    <div><strong>3D Engine:</strong> ${data.engine}</div>
                    <div><strong>C-Alpha Backbone Nodes Extracted:</strong> ${data.ca_coords.length}</div>
                `;
                jsonOut.textContent = JSON.stringify(data, null, 2);

                let pdbStr = '';
                data.ca_coords.forEach((c, i) => {
                    const idx = (i + 1).toString().padStart(5, ' ');
                    const resName = 'ALA';
                    const resIdx = (i + 1).toString().padStart(4, ' ');
                    const x = c.x.toFixed(3).padStart(8, ' ');
                    const y = c.y.toFixed(3).padStart(8, ' ');
                    const z = c.z.toFixed(3).padStart(8, ' ');
                    pdbStr += `ATOM  ${idx}  CA  ${resName} A${resIdx}    ${x}${y}${z}  1.00 20.00           C  \n`;
                });
                pdbStr += 'END\n';

                if (!viewer) {
                    viewer = await molstar.Viewer.create('viewport', {
                        layoutIsExpanded: false,
                        layoutShowControls: false,
                        viewportShowExpand: true
                    });
                }
                
                await viewer.loadStructureFromData(pdbStr, 'pdb');

            } catch (err) {
                knotOut.innerHTML = '<span style="color: #ef4444;">Bridge Error: Structural render failed (' + err.message + ').</span>';
            }
        }
    </script>
</body>
</html>"""

with open('index.html', 'w') as f:
    f.write(html_content)

print('[File Writer] index.html successfully generated with embedded Molstar WebGL viewer!')
