"""Wrapper to make mecab.Tagger compatible with python-mecab-ko API"""
import mecab

class MeCabWrapper:
    def __init__(self):
        self.tagger = mecab.Tagger('-d /opt/homebrew/lib/mecab/dic/mecab-ko-dic')
    
    def pos(self, text):
        """Parse text and return list of (word, pos) tuples"""
        result = []
        parsed = self.tagger.parse(text)
        for line in parsed.split('\n'):
            if line == 'EOS' or not line:
                break
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[0]
                features = parts[1].split(',')
                pos = features[0] if features else 'UNKNOWN'
                result.append((word, pos))
        return result

# Monkey patch for g2pkk
import sys
if 'mecab' not in sys.modules:
    sys.modules['mecab'] = type(sys)('mecab')
sys.modules['mecab'].MeCab = lambda: MeCabWrapper()
