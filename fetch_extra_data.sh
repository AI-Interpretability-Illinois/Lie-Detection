#!/bin/bash
# Fetch raw files for the extra evaluation sets (run on the compute node).
set -u
D=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
mkdir -p $D/ai_liar $D/mask $D/got $D/azaria
cd $D/ai_liar
for f in questions_1000_all.json wikidata_refined.json commonsense_QA_v2_dev.json; do
  curl -sfL --max-time 120 -o $f https://raw.githubusercontent.com/LoryPack/LLM-LieDetector/main/data/raw_questions/$f && echo "ok ai_liar/$f $(stat -c %s $f)"
done
curl -sfL --max-time 60 -o questions_loaders.py https://raw.githubusercontent.com/LoryPack/LLM-LieDetector/main/lllm/questions_loaders.py && echo "ok questions_loaders.py"
cd $D/mask
for c in known_facts provided_facts disinformation doubling_down_known_facts statistics continuations; do
  curl -sfL --max-time 120 -o $c.parquet "https://huggingface.co/datasets/cais/MASK/resolve/main/$c/test-00000-of-00001.parquet" && echo "ok mask/$c $(stat -c %s $c.parquet)"
done
cd $D/got
for c in cities neg_cities larger_than smaller_than sp_en_trans neg_sp_en_trans animal_class element_symb inventors facts companies_true_false; do
  curl -sfL --max-time 60 -o $c.csv https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/$c.csv && echo "ok got/$c $(wc -l < $c.csv) lines" || echo "missing got/$c"
done
echo FETCH_DONE
