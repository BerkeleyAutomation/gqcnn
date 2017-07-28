# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = GQCNN
SOURCEDIR     = source
BUILDDIR      = build
GH_PAGES_SOURCES = docs examples gqcnn tools

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: gh-pages
gh-pages:
	git checkout gh-pages && \
	cd .. && \
	git rm -rf . && git clean -fxd && \
	git checkout master $(GH_PAGES_SOURCES) && \
	git reset HEAD && \
	cd docs && \
	make html && \
	cd .. && \
	mv -fv docs/build/html/* ./ && \
	touch .nojekyll && \
	rm -rf $(GH_PAGES_SOURCES) && \
	git add -A && \
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && \
	git push origin --delete gh-pages && \
	git push origin gh-pages ; \
	git checkout master

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
