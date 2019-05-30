find . -name '*.py' | grep -v decoders/swig  | xargs git add
find . -name '*.sh' | grep -v decoders/swig  | xargs git add
git status
