/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "fixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "PatchFunction1.H"

//{{{ begin codeInclude
#line 62 "/home/bm424/OpenFOAM/bm424-11/run/sandTrap/sandTrap/0/epsilon/boundaryField/freeSurface"
#include "fvCFD.H"
				#include <cmath>
				#include <iostream>
//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = 5a6546cf365e3e78cb0ba69bbec5bd8f6a4e3d42
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void myEpsilonFS_5a6546cf365e3e78cb0ba69bbec5bd8f6a4e3d42(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchScalarField,
    myEpsilonFSFixedValueFvPatchScalarField
);

} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
myEpsilonFSFixedValueFvPatchScalarField::
myEpsilonFSFixedValueFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    parent_bctype(p, iF)
{
    if (false)
    {
        printMessage("Construct myEpsilonFS : patch/DimensionedField");
    }
}


Foam::
myEpsilonFSFixedValueFvPatchScalarField::
myEpsilonFSFixedValueFvPatchScalarField
(
    const myEpsilonFSFixedValueFvPatchScalarField& rhs,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    parent_bctype(rhs, p, iF, mapper)
{
    if (false)
    {
        printMessage("Construct myEpsilonFS : patch/DimensionedField/mapper");
    }
}


Foam::
myEpsilonFSFixedValueFvPatchScalarField::
myEpsilonFSFixedValueFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    parent_bctype(p, iF, dict)
{
    if (false)
    {
        printMessage("Construct myEpsilonFS : patch/dictionary");
    }
}


Foam::
myEpsilonFSFixedValueFvPatchScalarField::
myEpsilonFSFixedValueFvPatchScalarField
(
    const myEpsilonFSFixedValueFvPatchScalarField& rhs
)
:
    parent_bctype(rhs),
    dictionaryContent(rhs)
{
    if (false)
    {
        printMessage("Copy construct myEpsilonFS");
    }
}


Foam::
myEpsilonFSFixedValueFvPatchScalarField::
myEpsilonFSFixedValueFvPatchScalarField
(
    const myEpsilonFSFixedValueFvPatchScalarField& rhs,
    const DimensionedField<scalar, volMesh>& iF
)
:
    parent_bctype(rhs, iF)
{
    if (false)
    {
        printMessage("Construct myEpsilonFS : copy/DimensionedField");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
myEpsilonFSFixedValueFvPatchScalarField::
~myEpsilonFSFixedValueFvPatchScalarField()
{
    if (false)
    {
        printMessage("Destroy myEpsilonFS");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::
myEpsilonFSFixedValueFvPatchScalarField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        printMessage("updateCoeffs myEpsilonFS");
    }

//{{{ begin code
    #line 36 "/home/bm424/OpenFOAM/bm424-11/run/sandTrap/sandTrap/0/epsilon/boundaryField/freeSurface"
const fvPatch& boundaryPatch = this->patch();
			const volScalarField& eps = this->db().lookupObject<volScalarField>("epsilon");
			const fvMesh& mesh = eps.mesh();
			const vectorField& Cf = boundaryPatch.Cf();// face center coordinates for the patch
			const dimensionedScalar h = max(mesh.Cf().component(2));//flow depth in m
			fvPatchScalarField& epsilon = *this;

Info<<"depth: "<<h.value() << endl;

			const fvPatchScalarField& kinenp = boundaryPatch.lookupPatchField<volScalarField, scalar>("k");
			
			forAll(Cf, facei)
			{
				scalar kinenf = kinenp[facei];// turbulent kinetic energy at the facei
				epsilon[facei] = 2.3*pow(kinenf, 1.5) / h.value();
			}
//}}} end code

    this->parent_bctype::updateCoeffs();
}


// ************************************************************************* //

