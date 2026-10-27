"""Package the built HTML/JS game and its selected weights for GitHub Releases."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


root = Path(__file__).resolve().parents[1]
bundle = root / 'dist' / 'TrickyWords'
models = root / 'public' / 'models'
required = {'index.html', 'game.js', 'engine-asyncify.js', 'engine-jspi.js', 'engine-wasm.js',
            'READ-ME.txt', 'THIRD_PARTY.md', 'mobile-o-LICENSE.txt', 'birefnet-LICENSE.txt',
            'matter-LICENSE.txt', 'onnxruntime-LICENSE.txt', 'tokenizers-LICENSE.txt'}
assert required == {path.name for path in bundle.iterdir() if path.is_file()}, 'Build the complete download first.'
sources = [(bundle / name, f'TrickyWords/{name}') for name in sorted(required)]
manifest = json.loads((models / 'manifest.json').read_text())
cpu = json.loads((models / 'background-fp32-test.json').read_text())
weights = {}
for model in [*manifest['models'].values(), cpu]:
    for spec in [model, *model.get('externalData', [])]:
        name = spec['file']
        assert Path(name).name == name, 'Model paths must be filenames.'
        assert (models / name).stat().st_size == spec['bytes'], f'Wrong-size model: {name}'
        weights[name] = spec['bytes']
sources += [(models / name, f'TrickyWords/models/{name}') for name in sorted(weights)]
temporary = root / 'dist' / 'TrickyWords.zip.part'
output = root / 'dist' / 'TrickyWords.zip'
with ZipFile(temporary, 'w', ZIP_DEFLATED, compresslevel=1) as archive:
    for path, name in sources:
        archive.write(path, name)
with ZipFile(temporary) as archive:
    assert archive.testzip() is None, 'ZIP integrity check failed.'
    assert set(archive.namelist()) == {name for _, name in sources}
assert temporary.stat().st_size < 2 * 1024**3, 'The download exceeds the GitHub Releases per-asset limit.'
temporary.replace(output)
print(f'Created {output}: {output.stat().st_size / 1e9:.3f} GB, {len(sources)} files. ZIP integrity passed.')
