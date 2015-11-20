/**
 * \file        utilities.h
 * \author      Zongxiao He (zongxiah@bcm.edu)
 * \copyright   Copyright 2013, Leal Group
 * \date        2013-11-16
 *
 * \brief
 *
 *
 */


#ifndef UTILITIES_H
#define UTILITIES_H

#include <limits>
#include <iomanip>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <ctype.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdlib.h>
#include <json/json.h>


typedef std::vector<double> vectorF;
typedef std::vector< std::vector<double> > vector2F;

typedef std::vector<unsigned int> vectorUI;


namespace std {
//!- Dump a vector to screen
template<class T> ostream & operator<<(ostream & out, const vector<T> & vec)
  {
    if (!vec.empty()) {
      typename vector<T>::const_iterator it = vec.begin();
      out << setiosflags(ios::fixed) << setprecision(4) << setw(7) << *it;
      for (++it; it != vec.end(); ++it)
        out << " " << setiosflags(ios::fixed) << setprecision(4) << setw(7) <<  *it ;
    }
    return out;
  }
}

template<class T> Json::Value buildJsonArray (const std::vector<T> & vec)
{
    Json::Value jVec(Json::arrayValue);
    if (!vec.empty()) {
        typename std::vector<T>::const_iterator it = vec.begin();
        for( ; it != vec.end(); ++it){
            jVec.append(Json::Value(*it));
        }
    }
    return jVec;
}

template<class T> void resize_matrix( std::vector< std::vector<T> > & vect, int row, int col, T val)
{
	vect.resize(row, std::vector<T>(col, val));
	return;
}

template<class T> void reset_matrix( std::vector< std::vector<T> > & vect, int row, int col, T val)
{
	vect.clear();
	vect.resize(row, std::vector<T>(col, val));
	return;
}


template<class T> void copy_matrix(std::vector< std::vector<T> > & from, std::vector< std::vector<T> > & to, int row, int col)
{
	to.resize(row);
	for (int i = 0; i < row; ++i) {
		to[i].resize(col);
		std::copy(from[i].begin(), from[i].end(), to[i].begin());
	}
	return;
}


template<class T> void copy_vector(std::vector<T> & from, std::vector<T> & to, int len)
{
	to.resize(len);
	std::copy(from.begin(), from.end(), to.begin());
	return;
}


#endif /* UTILITIES_H */
