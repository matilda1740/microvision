import pandas as pd
import numpy as np
from pathlib import Path


def test_encode_templates_with_dummy_model(tmp_path):
    # import here to avoid heavy deps at test collection time
    from src.encode_chroma.encoder import encode_templates

    df = pd.DataFrame({"semantic_text": ["hello world", "goodbye"]})

    class DummyModel:
        def encode(self, batch, show_progress_bar=False, convert_to_numpy=True):
            # return deterministic small embeddings
            return np.vstack([[float(len(s)), 0.0] for s in batch])

    out_dir = tmp_path / "emb"
    embeddings, idx_df, collection = encode_templates(df, model=DummyModel(), output_dir=str(out_dir), chroma_dir=None, force_recompute=True)

    assert embeddings.shape[0] == 2
    assert isinstance(idx_df, pd.DataFrame)
    assert collection is None
    assert (out_dir / "embeddings.npy").exists()
    assert (out_dir / "embeddings_index.csv").exists()
