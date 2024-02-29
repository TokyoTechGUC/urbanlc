cd docs
make html
cd ..
python -m http.server -b localhost -d docs/build/html/