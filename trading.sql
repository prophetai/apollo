--
-- PostgreSQL database dump
--

-- Dumped from database version 12.4
-- Dumped by pg_dump version 12.2

-- Started on 2020-08-26 22:11:14 CDT

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 202 (class 1259 OID 24577)
-- Name: historical_usdjpy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.historical_usdjpy (
    date timestamp with time zone,
    "openBid" double precision,
    "openAsk" double precision,
    "highBid" double precision,
    "highAsk" double precision,
    "lowBid" double precision,
    "lowAsk" double precision,
    "closeBid" double precision,
    "closeAsk" double precision,
    volume bigint,
    complete boolean,
    id integer NOT NULL
);


ALTER TABLE public.historical_usdjpy OWNER TO postgres;

--
-- TOC entry 204 (class 1259 OID 24586)
-- Name: historical_usdjpy_ID_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public."historical_usdjpy_ID_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public."historical_usdjpy_ID_seq" OWNER TO postgres;

--
-- TOC entry 3145 (class 0 OID 0)
-- Dependencies: 204
-- Name: historical_usdjpy_ID_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public."historical_usdjpy_ID_seq" OWNED BY public.historical_usdjpy.id;


--
-- TOC entry 3009 (class 2604 OID 24588)
-- Name: historical_usdjpy id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historical_usdjpy ALTER COLUMN id SET DEFAULT nextval('public."historical_usdjpy_ID_seq"'::regclass);


--
-- TOC entry 3146 (class 0 OID 0)
-- Dependencies: 204
-- Name: historical_usdjpy_ID_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public."historical_usdjpy_ID_seq"', 1, true);


--
-- TOC entry 3011 (class 2606 OID 24593)
-- Name: historical_usdjpy historical_usdjpy_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historical_usdjpy
    ADD CONSTRAINT historical_usdjpy_pkey PRIMARY KEY (id);


-- Completed on 2020-08-26 22:11:15 CDT

--
-- PostgreSQL database dump complete
--