import sys
import os
import requests

SERVER_URL = "http://127.0.0.1:8000/predict"

def test_sequence(fasta_or_seq):
    # Strip FASTA headers if present
    lines = fasta_or_seq.strip().splitlines()
    sequence = "".join([l.strip() for l in lines if not l.startswith(">")])

    print(f"\n[CLI Tester] Input Sequence Length: {len(sequence)} AA")
    print(f"[CLI Tester] Sending request to {SERVER_URL} ...")

    try:
        response = requests.post(
            SERVER_URL,
            json={"sequence": sequence},
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        if response.status_code == 200:
            data = response.json()
            coords = data.get("ca_coords", [])
            print("=" * 60)
            print("  SUCCESS: BACKEND C-ALPHA EXTRACTION COMPLETE")
            print("=" * 60)
            print(f"  Engine Used       : {data.get('engine')}")
            print(f"  Sequence Length   : {data.get('sequence_length')} AA")
            print(f"  Extracted C-alphas: {len(coords)} nodes")
            if coords:
                print(f"  First Node (C-a0) : {coords[0]}")
                print(f"  Last Node  (C-aN) : {coords[-1]}")
            print("=" * 60)
        else:
            print(f"[CLI Tester Error] Server responded with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[CLI Tester Error] Failed to connect to server: {e}")

if __name__ == "__main__":
    default_seq = (
        "MSEFRAQWRNPGDVFSVLLIVGGDVILLALACVTGRVFTPVAFSFGWVAYAISAVVAAVGDNKLLARCTP"
        "EIPLKVINLKNGYNRDNKSWLLGRFMKDYNYWMPQDVRERLVRPPVRPDEEDRVAHDPFPARSSGLEPFP"
        "TNVTMGRSNAALCITVYRWVDGAKLGLPDRDWAWWSGFFMSILQLGVAAIPWGLHNNWNIFIATAAGTVM"
        "AYASASLPQWQIEKWTPERRKKDIAITQGNGNQHVIVILGADHGFDMEALAAARSPELGSTRIYTSILAF"
        "LWLVLLVTCCGIETDTWYLIAVGGLGMVQNLVVAGVPRKPAALGLPIELDRNLKDGQAAPNIFAEPKVMW"
        "SIMEFEKRFPGCGNALVKEFFPGDLLEWEVSWWASEDLAEREGLLSTQRRKAWDKAYEAAQEGR"
    )

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.exists(arg):
            with open(arg, "r") as f:
                test_sequence(f.read())
        else:
            print(f"[CLI Tester] Input '{arg}' is not a valid file path. Processing as raw sequence string...")
            test_sequence(arg)
    else:
        print("[CLI Tester] No input sequence provided. Running default unknotted baseline sequence...")
        test_sequence(default_seq)
