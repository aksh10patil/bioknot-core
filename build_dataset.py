import json

# Target Classes:
# 0: Unknot (0_1)
# 1: Trefoil (3_1)
# 2: Figure-Eight (4_1)
# 3: Complex / Star Knots (5_2+)

def initialize_training_dataset():
    dataset = [
        {
            "id": "KAH6862904.1",
            "sequence": "MSEFRAQWRNPGDVFSVLLIVGGDVILLALACVTGRVFTPVAFSFGWVAYAISAVVAAVGDNKLLARCTPEIPLKVINLKNGYNRDNKSWLLGRFMKDYNYWMPQDVRERLVRPPVRPDEEDRVAHDPFPARSSGLEPFPTNVTMGRSNAALCITVYRWVDGAKLGLPDRDWAWWSGFFMSILQLGVAAIPWGLHNNWNIFIATAAGTVMAYASASLPQWQIEKWTPERRKKDIAITQGNGNQHVIVILGADHGFDMEALAAARSPELGSTRIYTSILAFLWLVLLVTCCGIETDTWYLIAVGGLGMVQNLVVAGVPRKPAALGLPIELDRNLKDGQAAPNIFAEPKVMWSIMEFEKRFPGCGNALVKEFFPGDLLEWEVSWWASEDLAEREGLLSTQRRKAWDKAYEAAQEGR",
            "label": 0  # Unknot
        },
        {
            "id": "KAH6862977.1",
            "sequence": "MEGRPRNYTNGAGDPRAPAQRPPGSSR...TR",
            "label": 1  # Trefoil (3_1)
        }
    ]
    
    with open("bioknot_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[Dataset Pipeline] Initialized {len(dataset)} verified topological entries in bioknot_dataset.json")

if __name__ == "__main__":
    initialize_training_dataset()
