.PHONY:	all clean tests test_context test_mem test_conv test_weights

all:	tests

test_context:
	$(MAKE) -C test_context $@

test_mem:
	$(MAKE) -C test_mem $@

test_conv:
	$(MAKE) -C test_conv $@

test_weights:
	$(MAKE) -C test_weights $@

tests:	test_context test_mem test_conv test_weights

clean:
	$(MAKE) -C test_context $@
	$(MAKE) -C test_mem $@
	$(MAKE) -C test_conv $@
	$(MAKE) -C test_weights $@
