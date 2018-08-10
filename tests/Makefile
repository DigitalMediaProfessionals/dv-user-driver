.PHONY:	all clean tests test_context test_mem test_weights test_conv test_fc

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

tests:	test_context test_mem test_weights test_conv test_fc

clean:
	$(MAKE) -C test_context $@
	$(MAKE) -C test_mem $@
	$(MAKE) -C test_weights $@
	$(MAKE) -C test_conv $@
	$(MAKE) -C test_fc $@
