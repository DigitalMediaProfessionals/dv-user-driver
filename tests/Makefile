.PHONY:	all clean tests test_context test_mem test_weights test_conv test_fc test_lrn test_pool test_upsampling test_multirun test_maximizer

all:	tests

test_context:
	$(MAKE) -C test_context $@

test_mem:
	$(MAKE) -C test_mem $@

test_weights:
	$(MAKE) -C test_weights $@

test_conv:
	$(MAKE) -C test_conv $@

test_fc:
	$(MAKE) -C test_fc $@

test_lrn:
	$(MAKE) -C test_lrn $@

test_pool:
	$(MAKE) -C test_pool $@

test_upsampling:
	$(MAKE) -C test_upsampling $@

test_multirun:
	$(MAKE) -C test_multirun $@

test_maximizer:
	$(MAKE) -C test_maximizer $@

tests:	test_context test_mem test_weights test_conv test_fc test_lrn test_pool test_upsampling test_multirun test_maximizer

clean:
	$(MAKE) -C test_context $@
	$(MAKE) -C test_mem $@
	$(MAKE) -C test_weights $@
	$(MAKE) -C test_conv $@
	$(MAKE) -C test_fc $@
	$(MAKE) -C test_lrn $@
	$(MAKE) -C test_pool $@
	$(MAKE) -C test_upsampling $@
	$(MAKE) -C test_multirun $@
	$(MAKE) -C test_maximizer $@
