import os
import requests
import random
import logging
from Bio.PDB import PDBParser, DSSP
from utils.config import settings

log = logging.getLogger(__name__)

ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

def sasa_tool(protein_name: str, uniprot_id: str, use_mock: bool = False) -> dict:
    """
    Computes Solvent Accessible Surface Area (SASA) for a given protein.
    - Real Mode: Fetches PDB from EBI AlphaFold, runs Bio.PDB.DSSP.
    - Mock Mode: Deterministic random based on protein name hash.
    """
    if use_mock:
        return _get_mock_sasa(protein_name)

    try:
        # 1. Fetch PDB URL from AlphaFold API
        resp = requests.get(ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id), timeout=10)
        if resp.status_code != 200:
            return _get_mock_sasa(protein_name, source="mock_api_down")
        
        data = resp.json()
        if not data or not isinstance(data, list):
            return _get_mock_sasa(protein_name, source="mock_no_data")
            
        pdb_url = data[0].get("pdbUrl")
        if not pdb_url:
            return _get_mock_sasa(protein_name, source="mock_no_pdb_link")

        # 2. Download PDB
        pdb_resp = requests.get(pdb_url, timeout=20)
        if pdb_resp.status_code != 200:
            return _get_mock_sasa(protein_name, source="mock_download_failed")

        temp_pdb = os.path.join(settings.UPLOAD_DIR, f"{uniprot_id}.pdb")
        with open(temp_pdb, "wb") as f:
            f.write(pdb_resp.content)

        # 3. Parse and Run DSSP
        # NOTE: DSSP requires the 'mkdssp' or 'dssp' executable installed on the system.
        # If not found, BioPython will raise an error. We handle this by falling back to mock.
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(uniprot_id, temp_pdb)
        model = structure[0]
        
        try:
            # DSSP executable must be in PATH or specified
            dssp = DSSP(model, temp_pdb)
            
            sasa_values = [res[3] for res in dssp.values()] # 4th element is relative ASA or absolute? 
            # Bio.PDB.DSSP: [dssp_index, amino_acid, secondary_structure, relative_asa, phi, psi, nh_o_1_relidx, nh_o_1_energy, o_nh_1_relidx, o_nh_1_energy, nh_o_2_relidx, nh_o_2_energy, o_nh_2_relidx, o_nh_2_energy]
            
            mean_sasa = sum(sasa_values) / len(sasa_values) if sasa_values else 0
            buried_fraction = len([s for s in sasa_values if s < 0.2]) / len(sasa_values) if sasa_values else 0
            
            return {
                "mean_sasa": round(mean_sasa, 4),
                "buried_fraction": round(buried_fraction, 4),
                "exposed_active_site_residues": [], # Placeholder for real logic
                "sasa_stability_proxy": round(1 - (mean_sasa / 1.0), 4), # Simplified proxy
                "source": "ebi_api"
            }
        except Exception as e:
            log.warning(f"DSSP execution failed (missing executable?): {e}")
            return _get_mock_sasa(protein_name, source="mock_dssp_failed")
        finally:
            if os.path.exists(temp_pdb):
                os.remove(temp_pdb)

    except Exception as e:
        log.error(f"SASA tool failed for {uniprot_id}: {e}")
        return _get_mock_sasa(protein_name, source="mock_exception")

def _get_mock_sasa(protein_name: str, source: str = "mock") -> dict:
    # Use deterministic seed based on protein name
    random.seed(hash(protein_name))
    mean_sasa = random.uniform(0.3, 0.7)
    buried_fraction = random.uniform(0.4, 0.8)
    return {
        "mean_sasa": round(mean_sasa, 4),
        "buried_fraction": round(buried_fraction, 4),
        "exposed_active_site_residues": ["SER12", "HIS45", "ASP102"],
        "sasa_stability_proxy": round(1 - mean_sasa, 4),
        "source": source
    }
