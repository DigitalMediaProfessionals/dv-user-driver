.PHONY:	all clean tests test_context test_mem test_cmdlist test_weights

all:	tests

test_context:
	$(MAKE) -C test_context $@

test_mem:
	$(MAKE) -C test_mem $@

test_cmdlist:
	$(MAKE) -C test_cmdlist $@

test_weights:
	$(MAKE) -C test_weights $@

tests:	test_context test_mem test_cmdlist test_weights

clean:
	$(MAKE) -C test_context $@
	$(MAKE) -C test_mem $@
	$(MAKE) -C test_cmdlist $@
	$(MAKE) -C test_weights $@
