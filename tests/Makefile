.PHONY:	all clean tests test_mem test_cmdlist

all:	tests

test_mem:
	$(MAKE) -C test_mem $@

test_cmdlist:
	$(MAKE) -C test_cmdlist $@

tests:	test_mem test_cmdlist

clean:
	$(MAKE) -C test_mem $@
	$(MAKE) -C test_cmdlist $@
