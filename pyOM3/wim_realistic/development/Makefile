# Reference build of the ecwam IPHYS=0 source routines (unchanged ecwam
# code) as an f2py module, for wim_realistic/development/janssen_ref_test.py.
#
#   git clone --depth 1 https://github.com/ecmwf-ifs/ecwam.git   (Apache-2)
#   make ECWAM=/path/to/ecwam
#
# yow_stubs.F90 replaces the ecwam/fiat modules the routines USE (only the
# variables they read), dummies.F90 the routines behind the options that are
# off (gustiness, capped Charnock, gravity-capillary z0), wrap.F90 carries
# explicit-dimension entry points for f2py. The *.intfb.h files are the
# interface blocks ecwam generates at build time.
ECWAM ?= ../../../../ecwam
SRC    = $(ECWAM)/src/ecwam
ROUT   = init_x0tauhf sinput_jan sdissip_jan tau_phi_hf stresso z0wave fkmean femeanws frcutindex imphftail
FILES  = $(addsuffix .F90,$(ROUT))
ONLY   = w_setup w_get_tauhf_table w_sinput_jan w_sdissip_jan w_fkmean w_femeanws w_frcutindex w_stresso w_tau_phi_hf w_z0wave w_imphftail

all: $(FILES)
	python3 -m numpy.f2py -c -m ecwref yow_stubs.F90 dummies.F90 $(FILES) wrap.F90 \
	   only: $(ONLY) : --f90flags="-cpp -I$(CURDIR) -ffree-line-length-none"

%.F90: $(SRC)/%.F90
	cp $< $@

clean:
	rm -f ecwref*.so $(FILES)
