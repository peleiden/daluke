CC=gcc
CFLAGS=-fPIC -Wall -O3
LDFLAGS=-shared

all: so/collate.so

daluke/pretrain/data/collate.o: daluke/pretrain/data/collate.c

so/collate.so: daluke/pretrain/data/collate.o
	$(CC) daluke/pretrain/data/collate.o -shared -fPIC -o so/collate.so
	$(RM) daluke/pretrain/data/*.o

clean:
	$(RM) daluke/pretrain/data/collate/*.o so/*.so
